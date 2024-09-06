"""
The file has been adapted from the following files:
https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/experts.py
 Git commit hash: f3943cf9109226ed3ecf2d5dbb639a11cd925555
 We retain the following license from the original files:
"""
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.modules.mlp import new_feed_forward
from internlm.utils.common import get_current_device
from internlm.utils.logger import get_logger

from .base_layer import BaseMoELayer
from .utils import all_to_all

internlm_accelerator = get_accelerator()

try:
    # To enable gemm permute optimizations on GPU:
    #   python3 -m pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.3
    import grouped_gemm

    GEMM_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    # Fail silently so we don't spam logs unnecessarily if user isn't using gemm
    GEMM_INSTALLED = False
    pass

# global llm logger
logger = get_logger(__file__)

internlm_accelerator = get_accelerator()

uniform_map: Dict[torch.device, Callable] = {}


def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - epsilon, device=device), high=torch.tensor(1.0 + epsilon, device=device)
        ).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)


def custom_argsort(x, stable=True):
    if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:
        sorted_indices = torch.sort(x.to(torch.float), stable=stable)[1]
        return sorted_indices
    else:
        return torch.argsort(x, stable=stable)


class TopKGate(Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::
        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)
    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf
    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        topk: int = 1,
        noisy_gate_policy: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Deepspeed's mechisms, alway use fp32
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.k = topk

        self.noisy_gate_policy = noisy_gate_policy

    def forward(self, inputs: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        # input jittering
        if self.noisy_gate_policy == "Jitter" and self.training:
            inputs = multiplicative_jitter(inputs, device=inputs.device)
        logits = self.wg(inputs)
        gates = F.softmax(logits, dim=1)

        return gates


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity

class moe_gather(torch.autograd.Function):
    """Gather the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_):
        """Gather the input tensor based on the map tensor."""
        ctx.input_size = input_.size()
        ctx.map = map_
        return torch.gather(input_, 0, map_)

    @staticmethod
    def backward(ctx, grad_output):
        """Scatter the grad_output tensor based on the map tensor."""
        input_size = ctx.input_size
        map_ = ctx.map

        output = torch.zeros(
            input_size, dtype=grad_output.dtype, device=torch.cuda.current_device()
        )
        output.scatter_add_(0, map_, grad_output)
        return output, None, None


class moe_scatter(torch.autograd.Function):
    """Scatter the input tensor based on the map tensor."""

    @staticmethod
    def forward(ctx, input_, map_, output_size=None):
        """Scatter the input tensor based on the map tensor."""
        ctx.map = map_
        if output_size is not None:
            output = torch.zeros(output_size, dtype=input_.dtype, device=input_.device)
        else:
            output = torch.zeros_like(input_)

        output.scatter_add_(0, map_, input_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Gather the grad_output tensor based on the map tensor."""
        map_ = ctx.map
        grad_input = torch.gather(grad_output, 0, map_)
        return grad_input, None, None, None


def _gather_along_first_dim_moe(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = gpc.get_group(ParallelMode.EXPERT)
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=get_current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output

def _reduce_scatter_along_first_dim_moe(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    group = gpc.get_group(ParallelMode.EXPERT)
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce_scatter_along_first_dim_moe(grad_output), None


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):
        """Forward function."""
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_first_dim_moe(grad_output), None


def gather_from_parallel_region_to_moe(input_):
    """Wrapper for autograd function"""
    return _GatherFromSequenceParallelRegionToMOE.apply(input_)


def reduce_scatter_to_parallel_region_from_moe(input_):
    """Wrapper for autograd function"""
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)


class DroplessMoELayer(BaseMoELayer):
    """MoELayer module which implements MixtureOfExperts as described in Gshard_."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        activation_type: str = "swiglu",
        drop_and_pad: bool = False,
        drop_policy="probs",
        capacity_factor: float = None,
        noisy_gate_policy: str = None,
        moe_grouped_mlp: bool = True,
        use_test_mlp: bool = False,
        enable_fused_permute: bool = True,
    ) -> None:
        assert noisy_gate_policy is None or noisy_gate_policy in ["None", "Jitter", "RSample"], (
            "Unsupported noisy_gate_policy: " + noisy_gate_policy
        )
        assert (
            num_experts % ep_size == 0
        ), f"Number of experts ({num_experts}) should be divisible by expert parallel size ({ep_size})"

        if moe_grouped_mlp:
            assert False, "not support yet"
        else:
            experts = torch.nn.ModuleList(
                [
                    new_feed_forward(
                        in_features,
                        hidden_features,
                        out_features,
                        bias=False,
                        device=device,
                        dtype=dtype,
                        mlp_layer_fusion=mlp_layer_fusion,
                        multiple_of=multiple_of,
                        activation_type=activation_type,
                        is_expert=True,
                        use_test=use_test_mlp,
                    )
                    for _ in range(num_experts // ep_size)
                ]
            )
        super().__init__(
            TopKGate(
                in_features,
                num_experts,
                top_k,
                noisy_gate_policy,
            ),
            experts,
            ep_group,
            ep_size,
            num_experts // ep_size,
        )

        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        local_expert_indices_offset = gpc.get_local_rank(ParallelMode.EXPERT) * self.num_local_experts
        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        self.topk = top_k
        self.moe_grouped_mlp = moe_grouped_mlp
        local_expert_indices_offset = gpc.get_local_rank(ParallelMode.EXPERT) * self.num_local_experts
        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"

        # self.local_probs: probs of global token assignment to local experts.
        self.local_probs = None
        self.capacity_factor = None

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where
        # each element is True if it's between the local_expert_indices. Only useful when cross
        # device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

    def forward(self, *inputs: Tensor) -> Tensor:
        self.hidden_shape = inputs[0].shape

        d_model = inputs[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        reshaped_inputs = inputs[0].reshape(-1, d_model)

        self.gates = self.gate(reshaped_inputs)
        expert_weights, indices = self.topk_softmax_with_capacity(self.gates)

        (dispatched_input, tokens_per_expert) = self.token_permutation(reshaped_inputs, expert_weights, indices)
        if self.moe_grouped_mlp:
            expert_output = self.experts(dispatched_input, batch_sizes=tokens_per_expert)
        else:
            expert_output = self.experts(dispatched_input, split_size_or_sections=tokens_per_expert, split_dim=0)
        output, _ = self.token_unpermutation(expert_output, expert_weights)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output

    def topk_softmax_with_capacity(self, gates):
        expert_weights, indices = torch.topk(gates, self.topk, dim=1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)

        # without capacity
        if self.capacity_factor is None:
            # shape: [num_token, topk]
            return expert_weights, indices

        # with capacity
        expert_capacity = get_capacity(
            num_tokens=gates.shape[0] * self.topk,
            num_experts=gates.shape[1],
            capacity_factor=self.capacity_factor,
        )
        # TopK selection, Maskout unused experts
        topk_masked_gates = torch.zeros_like(gates).scatter(1, indices, expert_weights)
        topk_mask = torch.zeros_like(gates).scatter(1, indices, 1)
        if self.drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
        elif self.drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(gates).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {self.drop_policy}")
        if self.drop_and_pad:
            # shape: [num_expert, capacity]
            final_expert_weights, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, indices)
            # shape: [num_token, topk]
            final_expert_weights = expert_weights * torch.logical_not(exceed_mask)
            final_indices = indices.clone().masked_fill_(exceed_mask, torch.iinfo(torch.long).max)

        return final_expert_weights, final_indices

    def permute(self, tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
        """Permute the tokens based on the indices. Token with the same index will be grouped together.
        The input indices shape is [tokens, top_k], it indicates which experts were selected by each token separately.
        Args:
            tokens (torch.Tensor): The input token tensor.
            indices (torch.Tensor): The token to expert indices tensor, should have a shape of [num_tokens] or
                        [num_tokens, topk].
            num_out_tokens (int, optional): The effective output token count, when enabling the capacity factor, should
                        equal the number of tokens not dropped.  By default, set to None, meaning no tokens are dropped.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity]
                        to denote selected tokens per expert. Defaults to False.
        Returns:
            torch.Tensor: The permuted tensor.
            torch.Tensor: The sorted_indices corresponding permuted tensor.
        """
        if padded_mode:
            return self.permute_with_padded_tokens(tokens, indices)

        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = custom_argsort(flatten_indices, stable=True)

        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices

    def token_permutation(
        self, hidden_states: torch.Tensor, max_prob: torch.Tensor, max_ind: torch.Tensor
    ):
        """Dispatch tokens to local experts. It's composed of two stages:
        (1) Permute the tokens across the expert parallel devices. After this stage,
        each device receives all of the tokens assigned to its local set of experts
        in its local HBM.
        (2) Permute the tokens locally so that they are grouped by their expert
        assignment. After the stage (1), the tokens are grouped by which device
        they came from. We re-order them locally for subsequent efficient computation.

        Args:
            hidden_states: 3D tensor [S/TP, B, H]. Input tokens.
            max_prob: 2D tensor [S/TP*B, topk]. Each row of max_prob contains
            the probility distribution across `topk` experts for one local token.
            For 'aux_loss' load balancing, the sum of the values in each row is 1,
            thus for `top1` gating, it degenerates into a full 1 tensor.
            max_ind: 2D tensor [num_local_tokens, topk], where
            `num_local_tokens=S/TP*B`. Token assignment to global experts.

        Returns:
            permuted_local_hidden_states: Permutation of tokens to local experts group.
            tokens_per_expert: the number of tokens each local expert to process.
        """

        # Permute the tokens across the expert parallel devices.
        if self.ep_size > 1:
            ## local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, topk] -> [num_global_tokens, topk], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                global_indices = gather_from_parallel_region_to_moe(max_ind)
                # Create a mask of mapping between global and local tokens where each
                # element is True if it's between the local_expert_indices
                global_local_mask = (global_indices >= self.local_expert_indices[0]) & (
                    global_indices <= self.local_expert_indices[-1]
                )
                local_indices = global_indices.masked_select(global_local_mask)

            ## local_probs calculation
            # max_prob: [S/TP*B, topk] -> global_probs: [S*B*EP, topk]
            global_probs = gather_from_parallel_region_to_moe(max_prob)
            self.local_probs = global_probs.masked_select(global_local_mask)
            self.local_probs = self.local_probs.view(-1, 1)
            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            global_hidden_states = gather_from_parallel_region_to_moe(hidden_states) #, use_global_buffer=True
            # Reshape global_local_mask to be compatible with Tensor.gather
            global_local_map = global_local_mask.nonzero()[:, 0]
            self.global_local_map = global_local_map.view(-1, 1).expand(-1, hidden_states.shape[-1])
            local_hidden_states = moe_gather.apply(global_hidden_states, self.global_local_map)
        else:
            if self.topk > 1:
                global_local_mask = torch.ones_like(max_ind).bool()
                local_indices = max_ind.masked_select(global_local_mask)
                self.local_probs = max_prob.masked_select(global_local_mask)
                self.local_probs = self.local_probs.view(-1, 1)
                global_local_map = global_local_mask.nonzero()[:, 0]
                self.global_local_map = global_local_map.view(-1, 1).expand(
                    -1, hidden_states.shape[-1]
                )
                local_hidden_states = torch.gather(hidden_states, 0, self.global_local_map)
            else:
                local_indices = max_ind
                self.local_probs = max_prob.view(-1, 1)
                local_hidden_states = hidden_states
                self.global_local_map = None

        with torch.no_grad():
            tokens_per_expert = torch.bincount(
                local_indices.view(-1), minlength=self.num_experts
            )
            if self.num_local_experts < self.num_experts:
                tokens_per_expert = tokens_per_expert[
                    self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
                ]
            tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        # Stage2: permute the tokens locally so that they are grouped by their expert assignment
        # Reshape indices to be compatible with Tensor.gather

        permuted_local_hidden_states, self.reversed_local_input_permutation_mapping = self.permute(
            local_hidden_states, local_indices
        )

        return permuted_local_hidden_states, tokens_per_expert
    
    def unpermute(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        expert_weights: torch.Tensor = None,
        padded_mode: bool = False,
        restore_shape: torch.Size = None,
    ):
        """Unpermute a tensor of permuted tokens based on sorted indices, and optionally merge the tokens
           with their corresponding expert_weights.
        Args:
            permuted_tokens (torch.Tensor): The tensor of permuted tokens to be unpermuted.
            sorted_indices (torch.Tensor): The tensor of sorted indices used to unpermute the tokens.
            expert_weights (torch.Tensor, optional): The tensor of expert_weights corresponding to the permuted tokens.
                        If provided, the unpermuted tokens will be merged with their respective expert_weights.
            padded_mode (bool, optional): If True, indicating the indices are padded to [num_expert, capacity]
                        to denote selected tokens per expert. Defaults to False.
            restore_shape (torch.Size, optional): The input shape before permutation, only used in padding mode.
                        Defaults to None.
        Returns:
            torch.Tensor: The unpermuted tokens, optionally merged with expert_weights.
        """
        if padded_mode:
            return self.unpermute_with_padded_tokens(
                permuted_tokens, sorted_indices, expert_weights, restore_shape=restore_shape
            )

        assert sorted_indices.numel() == permuted_tokens.size(0)
        if expert_weights is not None:
            # Unpermute and merge the tokens with their expert_weights
            num_unpermuted_tokens = expert_weights.numel()
            topk = expert_weights.size(1)
        else:
            # Unpermute the tokens without merge
            num_unpermuted_tokens = permuted_tokens.size(0)
            topk = 1

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=permuted_tokens.dtype,
            device=permuted_tokens.device,
        )
        unpermuted_tokens.index_put_((sorted_indices,), permuted_tokens, accumulate=False)
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
        if expert_weights is not None:
            unpermuted_tokens = unpermuted_tokens * expert_weights.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)

        return unpermuted_tokens


    def token_unpermutation(self, hidden_states: torch.Tensor, bias: torch.Tensor = None):
        """
        Reverse process of `dispatch()` which permutes the output of local
        experts locallay and across expert parallel rank into the original order to
        produce the final output.

        Args:
            hidden_states: 2D tensor [num_permuted_tokens_for_local_experts, H],
            output of local experts.
            bias (optional): The bias tensor.

        Returns:
            output_total: un-permuted updated hidden states output from all local experts
            with shape of [S/TP, B, H]
        """
        # Stage1: unpermute the tokens and bias locally respectively.
        # Scale the expert output prior to reduction and subsequent to local unpermutation if k > 1.

        unpermuted_local_hidden = self.unpermute(
            hidden_states, self.reversed_local_input_permutation_mapping
        )
        unpermuted_local_hidden = unpermuted_local_hidden * self.local_probs

        unpermuted_local_bias = None
        # if self.add_bias:
        #     assert bias is not None
        #     unpermuted_local_bias = torch.zeros_like(hidden_states)
        #     unpermuted_local_bias = self.unpermute(bias, self.reversed_local_input_permutation_mapping)
        #     unpermuted_local_bias = unpermuted_local_bias * self.local_probs

        output_total = unpermuted_local_hidden
        output_bias_total = unpermuted_local_bias

        # Unpermute the tokens across expert parallel devices.
        if self.ep_size > 1:
            assert (
                self.global_local_map is not None
            ), "global_local_map is necessary for `AllGather`."
            # hidden_shape: [S/TP, B, H], gloal_num_tokens = S/TP*B*(TP*EP)
            global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1] * self.ep_size
            global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
            assert self.global_local_map.shape == unpermuted_local_hidden.shape
            unpermuted_global_hidden = moe_scatter.apply(
                unpermuted_local_hidden, self.global_local_map, global_hidden_shape
            )
            output_total = reduce_scatter_to_parallel_region_from_moe(unpermuted_global_hidden)
            # if self.add_bias:
            #     # Unpermute the bias across expert parallel devices.
            #     unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
            #     unpermuted_global_bias = unpermuted_global_bias.scatter_add(
            #         0, self.global_local_map, unpermuted_local_bias
            #     )
            #     output_bias_total = (
            #         tensor_parallel.reduce_scatter_to_sequence_parallel_region_from_moe(
            #             unpermuted_global_bias
            #         )
            #     )
            #     # bias is duplicated across tensor parallelism ranks;
            #     # reduce scatter reduces bias across tensor parallel_ranks
            #     output_bias_total = (
            #         output_bias_total / parallel_state.get_tensor_model_parallel_world_size()
            #     )
        else:
            if self.topk > 1:
                global_num_tokens = self.hidden_shape[0] * self.hidden_shape[1]
                global_hidden_shape = [global_num_tokens, hidden_states.shape[-1]]
                unpermuted_global_hidden = torch.zeros(
                    global_hidden_shape,
                    dtype=hidden_states.dtype,
                    device=torch.cuda.current_device(),
                )
                output_total = unpermuted_global_hidden.scatter_add(
                    0, self.global_local_map, unpermuted_local_hidden
                )
                if self.add_bias:
                    unpermuted_global_bias = torch.zeros_like(unpermuted_global_hidden)
                    output_bias_total = unpermuted_global_bias.scatter_add(
                        0, self.global_local_map, unpermuted_local_bias
                    )

        # if self.add_bias:
        #     output_bias_total = output_bias_total.view(self.hidden_shape)
        # else:
        #     output_bias_total = None

        return output_total, output_bias_total


    def load_balancing_loss(self, num_local_tokens_per_expert, gates):
        """Calculate the load balancing loss contribution."""
        assert len(gates.size()) == 2
        tokens, num_experts = gates.size()
        assert num_experts == self.num_experts
        assert len(num_local_tokens_per_expert.size()) == 1
        (num_experts,) = num_local_tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.topk)
        return scale * torch.dot(num_local_tokens_per_expert.to(gates.dtype), gates.mean(dim=0))
