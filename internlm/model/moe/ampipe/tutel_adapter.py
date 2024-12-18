import torch

from tutel.impls import losses
from tutel.impls.fast_dispatch import compute_sorted_location, GatingDecoder, GatingEncoder, TutelMoeFastDispatcher
from tutel.jit_kernels.gating import fast_cumsum_sub_one
from tutel.impls.communicate import simple_all_reduce

def extract_critical_encode(ctx, x, scores, top_k, loss_fn=losses.gshard_loss, capacity_factor=1.0, batch_prioritized_routing=False, normalize_gate=True, alignment=1, group=None, inequivalent_tokens=False):
    num_global_experts = int(scores.size(1))
    top_k, top_k_original = min(top_k, num_global_experts), top_k
    topk_indices = torch.topk(scores, top_k, dim=1).indices

    indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)]

    masks_se = [losses._one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype) for x in indices_s]
    ctx.scores0 = scores.detach()
    ctx.scores0.requires_grad = True
    with torch.enable_grad():
        gates_s = [(ctx.scores0 * x).sum(dim=1) for x in masks_se]
    ctx.gates_s = gates_s

    l_loss = loss_fn(scores, topk_indices) if loss_fn is not None else None

    if batch_prioritized_routing:
        importance_scores = -1 * scores.max(dim=1)[0]
        compute_location = lambda x: compute_sorted_location(x, importance_scores)
    else:
        compute_location = fast_cumsum_sub_one

    locations1 = compute_location(masks_se[0])

    locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

    if top_k > 1:
        acc_base = None
        for k in range(1, top_k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

        if normalize_gate:
            denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
            gates_s = [x / denom_s for x in gates_s]

    indices_s = [x.to(torch.int32) for x in indices_s]

    if inequivalent_tokens:
        num_samples = torch.tensor(scores.size(0), device=scores.device)
        num_samples = int(simple_all_reduce(num_samples, group=group, op=torch.distributed.ReduceOp.MAX))
    else:
        num_samples = int(scores.size(0))

    samples_per_expert = (num_samples + num_global_experts - 1) // num_global_experts
    if capacity_factor > 0:
        capacity = top_k * int(capacity_factor * samples_per_expert)
    else:
        capacity = torch.max(torch.cat(locations_s, dim=0))
        capacity = int(simple_all_reduce(capacity, group=group, op=torch.distributed.ReduceOp.MAX)) + 1
        if capacity_factor < 0:
            capacity = min(capacity, top_k * int(-capacity_factor * samples_per_expert))

    remainder = capacity % alignment
    if remainder > 0:
        capacity = capacity + alignment - remainder

    # if get_world_rank(group) == 0:
    #     logging.info(f"Capacity = {capacity}, real-time capacity-factor for top-{top_k_original} = {capacity / (top_k * samples_per_expert)}")

    crit = (num_global_experts, indices_s, locations_s, gates_s, capacity)
    top_experts = topk_indices

    tokens_per_expert = torch.histc(top_experts, bins=num_global_experts, min=0, max=num_global_experts)

    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, x.size(-1), x.dtype)
    dispatcher.update(indices_s, locations_s, gates_s, capacity, is_postscore=True)

    # assert dispatcher.dtype == torch.float16 and x.dtype == torch.float16 and torch.float16 == dispatcher.original_dtype
    x = GatingEncoder.forward(ctx, dispatcher, x)
    ctx.original_x_shape = x.size()
    return x.view(num_global_experts, -1, x.size(-1)), tokens_per_expert, dispatcher


def encode_bwd(ctx, grad_y):
    grad_y = grad_y.view(ctx.original_x_shape)
    grad_xs = GatingEncoder.backward(ctx, grad_y)
    return grad_xs[1]

def decode_fwd(ctx, x, dispatcher):
    #dispatcher.decode(x).view(-1, x.size(-1))
    # assert dispatcher.dtype == torch.float16 and x.dtype == torch.float16 and torch.float16 == dispatcher.original_dtype
    out = GatingDecoder.forward(ctx, dispatcher, x, *dispatcher.gates_)

    return out 

def decode_bwd(ctx, grad_y):
    grads = GatingDecoder.backward(ctx, grad_y)
    return grads[1], grads[2:] #scores grad