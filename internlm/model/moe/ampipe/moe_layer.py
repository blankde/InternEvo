from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import tutel
import megablocks_ops

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.moe.base_layer import BaseMoELayer
from internlm.model.moe.megablocks.mlp import MegaBlockFeedForward
from internlm.model.moe.utils import all_to_all

from internlm.model.moe.ampipe.tutel_adapter import extract_critical_encode, encode_bwd, decode_fwd, decode_bwd

try:
    from megablocks import ops
except (ModuleNotFoundError, ImportError):
    ops = None

import tutel.impls.communicate as C
import tutel_custom_kernel

class BWDDEBUG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, info):
        ctx.info = info 
        return inp

    @staticmethod
    def backward(ctx, grad_inp):
        if torch.distributed.get_rank() == 0:
            print("CALLING BWD: ", ctx.info)
        return grad_inp, None 



def create_fake(x):
    return megablocks_ops.fake_tensor(x)
    return x.detach() #if last line reports error

class NoBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        fake = create_fake(x) #watch out, do not access to x's data_ptr
        return fake
    @staticmethod
    def backward(ctx, g):
        return g

class NoBufferAssist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, assistant):
        #x.size() == assistant
        assert x.size() == assistant.size()
        assert assistant.requires_grad == False
        return assistant
    @staticmethod
    def backward(ctx, g):
        return g, None

class MLP_TP_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, group):
        ctx.group = group 
        return tokens
    @staticmethod
    def backward(ctx, g_tokens):
        torch.distributed.all_reduce(g_tokens, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return g_tokens, None

class MLP_TP_G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, group):
        torch.distributed.all_reduce(tokens, op=torch.distributed.ReduceOp.SUM, group=group)
        return tokens
    @staticmethod
    def backward(ctx, g_tokens):
        return g_tokens, None

_LOAD_BALANCING_LOSS = []

_MoE_Layer = []

import os
FAKE_A2A_SCALE=int(os.environ.get('FAKE_A2A_SCALE', 1))

def save_load_balancing_loss(loss, idx=-1):
    global _LOAD_BALANCING_LOSS
    if idx == -1 or len(_LOAD_BALANCING_LOSS) <= idx:
        _LOAD_BALANCING_LOSS.append(loss)
    else:
        _LOAD_BALANCING_LOSS[idx] = (_LOAD_BALANCING_LOSS[idx][0] + loss[0], torch.cat([_LOAD_BALANCING_LOSS[idx][1], loss[1]]))

def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()

def get_world_size(group=None):
    try:
        return torch.distributed.get_world_size(group)
    except:
        return 1

# def batched_load_balancing_loss(args : Arguments):
#     # tokens_per_expert[i].shape = (num_experts)
#     # expert_scores[i].shape = (tokens, num_experts)
#     tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
#     num_layers_per_pipeline_stage = (
#         gpc.config.model.num_layers // gpc.get_world_rank(ParallelMode.PIPELINE))
#     if args.num_layers_per_virtual_pipeline_stage is not None:
#         num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

#     if len(tokens_per_expert) != num_layers_per_pipeline_stage:
#         raise ValueError(
#             f"Expected {num_layers_per_pipeline_stage} token_per_experts "
#             f"but found {len(tokens_per_expert)}.\nnum_layers = "
#             f"{args.num_layers}\npipeline_model_parallel_size = "
#             f"{args.pipeline_model_parallel_size}\n"
#             "num_layers_per_virtual_pipeline_stage"
#             f" = {args.num_layers_per_virtual_pipeline_stage}")
#     if len(expert_scores) != num_layers_per_pipeline_stage:
#         raise ValueError(
#             f"Expected {num_layers_per_pipeline_stage} expert_scores "
#             f"but found {len(tokens_per_expert)}.\nnum_layers = "
#             f"{args.num_layers}\npipeline_model_parallel_size = "
#             f"{args.pipeline_model_parallel_size}\n"
#             "num_layers_per_virtual_pipeline_stage"
#             f" = {args.num_layers_per_virtual_pipeline_stage}")

#     # Verify the shape of the tokens_per_expert and expert_scores tensors.
#     assert all([
#         x.ndim == 1 and x.numel() == args.moe_num_experts
#         for x in tokens_per_expert
#     ])

#     tokens = expert_scores[0].shape[0]
#     assert all([
#         (x.ndim == 2 and x.shape[1] == args.moe_num_experts and
#          x.shape[0] == tokens) for x in expert_scores
#     ])


#     # Concatenate the contributions of each layer and convert to
#     # the correct types and formats for the dot product.
#     if args.moe_lbl_in_fp32:
#         expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
#     else:
#         expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)
#     tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

#     expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
#     assert tokens_per_expert.numel() == expected_values
#     assert expert_scores.numel() == expected_values

#     # Calculate the total scale across all factors.
#     #
#     # loss_weight * num_experts / (num_layers * tokens * top_k)
#     scale_numerator = (
#         args.moe_num_experts *
#         args.moe_loss_weight
#     )
#     scale_denominator = (
#         args.num_layers *
#         tokens *
#         args.moe_top_k
#     )
#     scale = scale_numerator / scale_denominator
#     return scale * torch.dot(tokens_per_expert, expert_scores)


class TopKGate(torch.nn.Module):
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

    def forward(self, inputs: torch.Tensor):
        # input jittering
        if self.noisy_gate_policy == "Jitter" and self.training:
            inputs = multiplicative_jitter(inputs, device=inputs.device)
        logits = self.wg(inputs)
        gates = F.softmax(logits, dim=1)

        return gates


class AmpipeMegaBlockMoE(BaseMoELayer):
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
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_experts: int,
        top_k: int,
        ep_group: Optional[torch.distributed.ProcessGroup],
        ep_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.device] = None,
        mlp_layer_fusion: bool = False,  # pylint: disable=W0613
        multiple_of: int = 256,
        activation_type: str = "swiglu",  # pylint: disable=W0613
        capacity_factor: float = 1.0,
        drop_tokens: bool = True,
    ) -> None:
        assert not gpc.config.parallel.sequence_parallel, "do not support sequence parallel"
        assert ops is not None, 'MegaBlocks not found, please run "pip install megablocks".'
        self.top_k = top_k
        self.num_experts = num_experts

        tp_size = gpc.get_world_size(ParallelMode.TENSOR)
        self.ffn_dim = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        assert self.ffn_dim % tp_size == 0
        super().__init__(
            TopKGate(
                in_features,
                num_experts,
                top_k,
            ),
            MegaBlockFeedForward(
                in_features,
                self.ffn_dim // tp_size,
                out_features,
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
        self.quantize_scatter_num_bits = -1
        # re-init the number of experts in each device
        self.num_local_experts = num_experts // ep_size

        self.forward_fn = self._parallel_forward if gpc.expert_parallel_size > 1 else self._forward

    def expert_capacity(self, tokens, top_k):
        world_size = gpc.get_world_size(ParallelMode.EXPERT)  # mpu.get_expert_parallel_world_size(self.args)
        tokens_per_expert = top_k * tokens * world_size / self.num_experts
        return int(self.capacity_factor * tokens_per_expert)

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
        bins = bins.view(1) if len(bins.size()) == 0 else bins
        return indices, bin_ids, bins, tokens_per_expert

    def _forward(self, x, expert_weights, top_experts) -> torch.Tensor:
        """
        x: (sequence_length, model_dim)
        gate_logits: (sequence_length, n_experts)
        """
        with torch.no_grad():
            indices, _, bins, tokens_per_expert = self.indices_and_bins(top_experts)
            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, _ = x.size()
            expert_capacity = self.expert_capacity(tokens, top_k=self.top_k)
            if not self.drop_tokens:
                expert_capacity = torch.max(tokens_per_expert).item()

        out = self.permute_and_compute(x, indices, expert_weights, bins, expert_capacity, top_k=self.top_k)

        return out, tokens_per_expert.flatten()

    def _parallel_forward(self, x, expert_weights, top_experts):
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
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = self.indices_and_bins(top_experts)

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
        x = x.view(-1, x.shape[-1])  # TODO can be deleted
        x = ops.gather(x, indices, bin_ids, bins, self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            experts_per_rank = self.num_local_experts  # mpu.experts_per_rank(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            world_size = gpc.get_world_size(ParallelMode.EXPERT)  # mpu.get_expert_parallel_world_size(self.args)
            tokens_per_expert = tokens_per_expert.view(
                world_size, experts_per_rank
            )  # ((1,2), (1,0)) in g1, ((2,0),(2,0)) in g2
            parallel_tokens_per_expert = parallel_tokens_per_expert.view(
                world_size, experts_per_rank
            )  # ((1,2), (2,0)) in g1, ((1,0),(2,0)) in g2

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = tokens_per_expert.cpu().sum(dim=-1)
            parallel_tokens_per_expert_cpu = parallel_tokens_per_expert.cpu()
            recv_counts = parallel_tokens_per_expert_cpu.sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

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

            replicate_bins = ops.inclusive_cumsum(parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = replicate_bins.view(1) if len(replicate_bins.size()) == 0 else replicate_bins

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(self.num_experts, dtype=torch.int32, device=indices.device),
                self.num_local_experts,  # mpu.experts_per_rank(self.args),
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0), replicate_bins, tokens_received
            ).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            _, parallel_indices = ops.sort(parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(parallel_tokens_per_expert, 0)
            parallel_bins = parallel_bins.view(1) if len(parallel_bins.size()) == 0 else parallel_bins

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, _ = x.size()
            expert_capacity = self.expert_capacity(tokens, top_k=1)
            if not self.drop_tokens:
                expert_capacity = torch.max(parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_indices,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1,
        )

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(parallel_x, send_counts, recv_counts, gpc.get_group(ParallelMode.EXPERT))

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k, self.quantize_scatter_num_bits)
        return x, tokens_per_expert.flatten()

    def permute_and_compute(self, x, indices, expert_weights, bins, expert_capacity, top_k):  # unused  # unused
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.binned_gather(x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation
        # First Dense x Dense -> Sparse for w1 and w3,
        # (top_k * sequence_length + padding, ffn_dim * n_experts)
        x = self.experts(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(x, indices, expert_weights, bins, top_k)

    def load_balancing_loss(self, tokens_per_expert, expert_scores):
        """Calculate the load balancing loss contribution."""
        assert len(expert_scores.size()) == 2
        tokens, num_experts = expert_scores.size()
        assert num_experts == self.num_experts
        assert len(tokens_per_expert.size()) == 1
        (num_experts,) = tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.top_k)
        return scale * torch.dot(tokens_per_expert.to(expert_scores.dtype), expert_scores.mean(dim=0))

    def dummy_moe_loss(self, *args, **kwargs):
        return None


    def parallel_forward_prepare(self, x, top_expert):
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
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_expert))

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(
                tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                tokens_per_expert,
                group=gpc.get_group(ParallelMode.EXPERT),
                async_op=True)

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # TODO(tgale): We can tune these kernels for this special case by
        # skipping the memset if tokens == padded_tokens and also taking
        # in an optional padded_tokens rather than copying it from the
        # device.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1])
        x = ops.gather(x, indices, bin_ids, bins, self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            world_size = gpc.get_world_size(ParallelMode.EXPERT)

            # Reshape to [world_size, num_experts_per_rank].
            tokens_per_expert = tokens_per_expert.view(world_size, -1)
            parallel_tokens_per_expert = (
                parallel_tokens_per_expert.view(world_size, -1))

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = tokens_per_expert.cpu().sum(dim=-1)
            recv_counts = parallel_tokens_per_expert.cpu().sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            replicate_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = (
                replicate_bins.view(1)
                if not len(replicate_bins.size())
                else replicate_bins
            )

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts, dtype=torch.int32, device=indices.device),
                self.num_local_experts,
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0),
                replicate_bins, tokens_received).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(
                parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1)
                if not len(parallel_bins.size())
                else parallel_bins
            )

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens, top_k=1)
            if not self.drop_tokens:
                expert_capacity = torch.max(parallel_tokens_per_expert).item()

        return x, recv_counts, send_counts, parallel_tokens_per_expert, \
            parallel_indices, parallel_bin_ids, parallel_bins, expert_capacity, \
            indices, bin_ids, bins, tokens_per_expert

    def parallel_forward_a2a1(self, x, recv_counts, send_counts):
        # Permute the tokens across the devices.
        parallel_x = all_to_all(
            x, recv_counts, send_counts,
            gpc.get_group(ParallelMode.EXPERT))
        return parallel_x
        # Locally permute the tokens and perform the expert computation.
    def parallel_forward_compute(self, parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            expert_capacity):

        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_indices,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1,
        )
        return parallel_x

    def parallel_forward_a2a2(self, parallel_x, send_counts, recv_counts):
        # Un-permute the tokens across the devices.
        x = all_to_all(
            parallel_x, send_counts, recv_counts,
            gpc.get_group(ParallelMode.EXPERT))
        return x 

    def parallel_forward_post(self, x, indices, bin_ids, bins, tokens_per_expert):

        # Un-permute locally to setup for the next series of operations.
        x = ops.scatter(x, indices, bin_ids, expert_weights, bins, self.top_k, self.quantize_scatter_num_bits)
        return x, tokens_per_expert.flatten()
    
    def tutel_prepare(self, x, scores):
        origin_shape = x.shape 
        x = x.view(-1, origin_shape[-1])
        crit, top_experts = tutel.tutel_moe.extract_critical(scores,
                top_k = self.args.moe_top_k,
                loss_fn = None,
                capacity_factor = self.capacity_factor
            )

        tokens_per_expert = ops.histogram(top_experts.view(-1), self.num_experts)

        y = tutel.tutel_moe.fast_encode(x.to(scores.dtype), crit, True).to(x.dtype)
        return y, tokens_per_expert, crit

    def tutel_a2a1(self, x):
        return tutel.impls.communicate.all_to_all(x, 1, 0, use_2dh=False, group=gpc.get_group(ParallelMode.EXPERT))
    
    def tutel_a2a2(self, x):
        return tutel.impls.communicate.all_to_all(x, 0, 1, use_2dh=False, group=gpc.get_group(ParallelMode.EXPERT))

    def tutel_post(self, x, crit, dtype):
        y = tutel.tutel_moe.fast_decode(x.to(dtype), crit, True)
        return y

    def hash_forward(self, x, timers, start, stop):
        pass 

    def tutel_prepare_fwd(self, ctx, x):
        ctx.x0 = x.detach()
        ctx.x0.requires_grad = True
        with torch.enable_grad():
            scores = self.gate(ctx.x0.view(-1, x.shape[-1]))
        ctx.scores = scores 
        origin_shape = x.shape 
        x = x.view(-1, origin_shape[-1])

        y, tokens_per_expert, dispatcher = extract_critical_encode(ctx, x, scores,
                top_k = self.top_k,
                loss_fn = None,
                capacity_factor = self.capacity_factor
            )
        return y, dispatcher, origin_shape, scores, tokens_per_expert 
        #y, crit, dispatcher = tutel.tutel_moe.fast_encode(x.to(scores.dtype), crit, True).to(x.dtype)


    def tutel_prepare_bwd(self, ctx, g_score, g_tokens, g_gates):
        
        grad_x = encode_bwd(ctx, g_tokens)
        for g_gate, gate in zip(g_gates, ctx.gates_s):
            gate.backward(g_gate)

        #print("score0:", ctx.scores0.grad)
        ctx.scores.backward(g_score + ctx.scores0.grad)
        #print("bwd: ", g_tokens.size(), grad_x.size(), ctx.x0.size(), ctx.x0.grad.size())
        grad_x = grad_x.view(ctx.x0.grad.size())
        return grad_x + ctx.x0.grad

    def tutel_mlp_fwd(self, ctx, tokens):
        ctx.tokens = tokens.detach()
        ctx.tokens.requires_grad = True
        with torch.enable_grad():
            y = self.experts(ctx.tokens)
            ctx.y = NoBuffer.apply(y)
        return y 

    def tutel_mlp_bwd(self, ctx, g_tokens):
        ctx.y.backward(g_tokens)
        return ctx.tokens.grad

    def tutel_a2a_scatter(self, tokens):
        # group = gpc.get_group(ParallelMode.EXPERT)
        world_size = gpc.get_world_size(ParallelMode.EXPERT) #world size not include TP ranks
        if world_size == 1:
            return tokens 
        
        # tokens = tokens.contiguous()
        # output = torch.empty_like(tokens)

        # C.AllToAllStatus.init(group, -1, -1)
        # tutel_custom_kernel.all_to_all_with_scale(tokens, output, FAKE_A2A_SCALE)
        '''
        torch.distributed.all_to_all_single(output, tokens, group=group)
        if FAKE_A2A_SCALE > 1:
            for i in range(FAKE_A2A_SCALE - 1):
                torch.distributed.all_to_all_single(output, tokens, group=group)
        '''
        output, _ = all_to_all(tokens, group=gpc.get_group(ParallelMode.EXPERT))

        output = output.view([world_size, -1] + list(output.shape[1:]))
        output = output.permute([1, 0] + list(range(2, output.dim())))
        output = output.contiguous().view(list(output.shape[:1]) + [-1] + list(output.shape[3:]))
        #y = tutel.impls.communicate.all_to_all(y, 1, 0, use_2dh=False, group=self.args.expert_parallel_group)
        return output 
    
    def tutel_a2a_scatter_p0(self, tokens):
        world_size = get_world_size(self.args.expert_parallel_group)
        if world_size == 1:
            return tokens 
        tokens = tokens.contiguous()
        output = torch.empty_like(tokens)
        return tokens, output 
    
    def tutel_a2a_scatter_p1(self, tokens, output):
        C.AllToAllStatus.init(self.args.expert_parallel_group, -1, -1)
        tutel_custom_kernel.all_to_all_with_scale(tokens, output, FAKE_A2A_SCALE)
    
    def tutel_a2a_scatter_p2(self, output):
        output = output.view([world_size, -1] + list(output.shape[1:]))
        output = output.permute([1, 0] + list(range(2, output.dim())))
        #print("o0.size: ", output.size()) #torch.Size([1, 8, 1280, 512])
        output = output.contiguous().view(list(output.shape[:1]) + [-1] + list(output.shape[3:]))
        return output 

    def tutel_a2a_gather(self, tokens):
        # group = gpc.get_group(ParallelMode.EXPERT)
        world_size = gpc.get_world_size(ParallelMode.EXPERT)
        if world_size == 1:
            return tokens 



        reshaped_input = tokens.view(list(tokens.shape[:1]) + [world_size, -1] + list(tokens.shape[2:]))
        reshaped_input = reshaped_input.permute([1, 0] + list(range(2, reshaped_input.dim()))).contiguous()
        #simple_all_to_all(reshaped_input, group, background=True)
        # local_input = torch.empty_like(reshaped_input)

        # C.AllToAllStatus.init(group, -1, -1)
        # tutel_custom_kernel.all_to_all_with_scale(reshaped_input, local_input, FAKE_A2A_SCALE)

        local_input, _ = all_to_all(reshaped_input, group=gpc.get_group(ParallelMode.EXPERT))

        '''        
        torch.distributed.all_to_all_single(local_input, reshaped_input, group=group)

        if FAKE_A2A_SCALE > 1:
            for i in range(FAKE_A2A_SCALE - 1):
                torch.distributed.all_to_all_single(local_input, reshaped_input, group=group)
        '''     
        local_input = local_input.view([-1] + list(local_input.shape[2:]))

        # if tp_info[0] > 1 :
        #     torch.distributed.all_reduce(local_input, op=torch.distributed.ReduceOp.SUM, group=tp_info[1])

        return local_input 
    
    def tutel_post_fwd(self, ctx, tokens, dispatcher):
        
        tokens = decode_fwd(ctx, tokens.view(-1, tokens.shape(-1)), dispatcher)

        return tokens

    def tutel_post_bwd(self, ctx, g_tokens):
        tokens_grad, scores_grad = decode_bwd(ctx, g_tokens)
        return tokens_grad, scores_grad

    def forward(self, *inputs) -> torch.Tensor:
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

        x, tokens_per_expert = self.forward_fn(x, expert_weights, top_experts)

        self.l_aux = self.dummy_moe_loss(tokens_per_expert, all_probs)

        return x.view(*input_shape)