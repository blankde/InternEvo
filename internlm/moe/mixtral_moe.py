from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.utils import Silu
from internlm.utils.registry import MODEL_INITIALIZER

from .base_moe import BaseMoELayer
from .utils import all_to_all


try:
    from megablocks import ops
except ImportError:
    print(
        "MegaBlocks not found, please see "
        "https://github.com/stanford-futuredata/megablocks/. "
        "Note that MegaBlocks depends on mosaicml-turbo, which only "
        "supports python 3.10."
    )


class TensorParallelBmm(torch.autograd.Function):
    """
    Tensor parallel sdd
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w, group=None):
        # [m, k] x [n, k] = [m, n]
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'x' and 'w'.")

        ctx.group = group
        ctx.save_for_backward(
            x,
            w,
        )

        return torch.bmm(x, w)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors[:2]

        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.bmm(grad, w.transpose(-2, -1))
        if ctx.group is not None:
            handle_x = torch.distributed.all_reduce(dx, group=ctx.group, async_op=True)

        dw = None
        if ctx.needs_input_grad[1]:
            dw = torch.bmm(x.transpose(-2, -1), grad)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        if ctx.group is not None:
            handle_x.wait()

        dw = dw.to(w.dtype)
        return dx, dw, None


def tensor_parallel_bmm(x, w, group=None):
    return TensorParallelBmm.apply(x, w, group)


class MLP(nn.Module):
    """
    Feed forward using megablock kernel
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_local_experts: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # merged expert weights, all of size  (ffn_dim * n_experts, model_dim)
        self.w1 = nn.Parameter(torch.empty(num_local_experts, in_features, hidden_features, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(num_local_experts, in_features, hidden_features, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(num_local_experts, hidden_features, in_features, device=device, dtype=dtype))

        # self.w1 = nn.Parameter(torch.load("w1.pt").to(device).view(num_local_experts, hidden_features, in_features).transpose(1,2).contiguous())
        # self.w2 = nn.Parameter(torch.load("w2.pt").to(device).view(num_local_experts, hidden_features, in_features).transpose(1,2).contiguous())
        # self.w3 = nn.Parameter(torch.load("w3.pt").to(device).view(num_local_experts, hidden_features, in_features).contiguous())

    def forward(self, x):
        #TODO w2 and w3 should swap
        w1_o = tensor_parallel_bmm(x, self.w1, group=gpc.get_group(ParallelMode.TENSOR))
        w2_o = tensor_parallel_bmm(x, self.w2, group=gpc.get_group(ParallelMode.TENSOR))
        out = tensor_parallel_bmm(Silu(w1_o, w2_o), self.w3)
        torch.distributed.all_reduce(out, group=gpc.get_group(ParallelMode.TENSOR))

        return out


@MODEL_INITIALIZER.register_module(module_name="Mixtral")
class MixtralMoE(BaseMoELayer):
    """
    Built on the paper and library Megablocks as described in
    https://arxiv.org/abs/2211.15841. This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(
        self,
        hidden_size,
        ep_group,
        ep_size,
        num_experts,
        top_k,
        device=None,
        dtype=None,
    ) -> None:
        assert not gpc.config.parallel.sequence_parallel, "do not support sequence parallel"
        assert gpc.config.parallel["tensor"].get("mode", "mtp") != "isp", "do not support weight parallel"
        self.top_k = top_k
        self.num_experts = num_experts

        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.ffn_dim = int(hidden_size * gpc.config.model.mlp_ratio) // ep_size
        self.moe_capacity_factor = 1
        assert self.ffn_dim % tp_size == 0
        super().__init__(
            torch.nn.Linear(hidden_size, num_experts, bias=False),
            MLP(
                hidden_size,
                self.ffn_dim // tp_size,
                num_experts // ep_size,
                device,
                dtype,
            ),
            ep_group,
            ep_size,
            1,
        )

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)
        # self.blocking = 128
        self.quantize_scatter_num_bits = -1

        # # Calculate the number of bits needed to represent the column indices
        # # in the intermediate sparse matrix.
        # max_column_index = (self.ffn_dim * self.num_experts) // self.blocking
        # self.transpose_sort_end_bit = max(int(np.ceil(np.log2(max_column_index))), 1)

        # re-init the number of experts in each device
        # self.num_local_experts = num_experts // ep_size

    def expert_capacity(self, tokens):
        world_size = gpc.get_world_size(ParallelMode.EXPERT)  # mpu.get_expert_parallel_world_size(self.args)
        tokens_per_expert = self.top_k * tokens * world_size / self.num_experts
        return int(self.moe_capacity_factor * tokens_per_expert)

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def _forward(self, *inputs) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        # optional reshape
        x = inputs[0]
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)

        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        weights, selected_experts = torch.topk(all_probs, self.top_k, dim=-1)
        weights /= weights.sum(dim=-1, keepdim=True)
        weights = weights.flatten().to(x.dtype)
        selected_experts = selected_experts.flatten()

        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (self.indices_and_bins(selected_experts))
            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(tokens_per_expert).item()

        out = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            weights,
            bins,
            expert_capacity,
            top_k=self.top_k
        )

        # print(out, flush=True)
        # exit(-1)
        return out

    def _parallel_forward(self, *inputs):
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        # optional reshape
        x = inputs[0]
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        # gate_logits: (sequence_length, n_experts)
        gate_logits = self.gate(x)

        # all_probs: (sequence_length, n_experts) and upcast for softmax
        all_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float)
        # weights, selected_experts: (sequence_length, top-k)
        expert_weights, top_experts = torch.topk(all_probs, self.top_k, dim=-1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (self.indices_and_bins(top_experts))

            # Pass token count information to the device on which the
            # target expert resides.
            # e.g. tokens_per_expert = (1,2,1,0) in g1
            #      tokens_per_expert = (2,0,2,0) in g2
            # then:parallel_tokens_per_expert = (1,2,2,0) in g1
            #      parallel_tokens_per_expert = (1,0,2,0) in g2
            parallel_tokens_per_expert = torch.empty_like(tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert, tokens_per_expert, group=gpc.get_group(ParallelMode.EXPERT), async_op=True
            )

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1]) #TODO can be deleted
        x = ops.gather(x, indices, bin_ids, bins, self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            experts_per_rank = self.num_local_experts  # mpu.experts_per_rank(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            world_size = gpc.get_world_size(ParallelMode.EXPERT)  # mpu.get_expert_parallel_world_size(self.args)
            tokens_per_expert = tokens_per_expert.view(world_size, experts_per_rank) # ((1,2), (1,0)) in g1, ((2,0),(2,0)) in g2
            parallel_tokens_per_expert = parallel_tokens_per_expert.view(world_size, experts_per_rank) # ((1,2), (2,0)) in g1, ((1,0),(2,0)) in g2

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        #
        # TODO(tgale): Fuse this into the prior, local permutation.
        # x = ops.repeat(x, (mpu.hidden_sharding_degree(self.args), 1))

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts, gpc.get_group(ParallelMode.EXPERT), async_op=True
        )

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.

            # ((1,2), (2,0)) in g1, ((1,0),(2,0)) in g2 -> (1,3,5,5) in g1, (1,1,3,3) in g2
            replicate_bins = ops.inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = replicate_bins.view(1) if not len(replicate_bins.size()) else replicate_bins

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(self.num_experts, dtype=torch.int32, device=indices.device),
                self.num_local_experts,  # mpu.experts_per_rank(self.args),
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
            ).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
            parallel_bins = parallel_bins.view(1) if not len(parallel_bins.size()) else parallel_bins

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            #if expert_capacity == 0:
            expert_capacity = torch.max(parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1
        )

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(parallel_x, send_counts, recv_counts, gpc.get_group(ParallelMode.EXPERT))

        # Reduce along the hidden sharding to get the final outputs.
        #
        # TODO(tgale): Fuse this into the following local permutation.
        shape = (1, -1, input_shape[1])  # mpu.hidden_sharding_degree(self.args),
        x = ops.sum(x.view(shape), dim=0)

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k, self.quantize_scatter_num_bits)
        # print(x, flush=True)
        # exit(-1)
        return x
    
    def permute_and_compute(
            self,
            x,
            tokens_per_expert, # unused
            indices,
            bin_ids, # unused
            expert_weights,
            bins,
            expert_capacity,
            top_k):
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = self.experts(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(x, indices, expert_weights, bins, top_k)

    def forward(self, *inputs) -> torch.Tensor:
        if gpc.expert_parallel_size > 1:
            return self._parallel_forward(*inputs)
        else:
            return self._forward(*inputs)
