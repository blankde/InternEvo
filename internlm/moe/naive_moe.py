import torch

from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.model.linear import FeedForward
from internlm.utils.registry import MODEL_INITIALIZER

from .base_moe import BaseMoELayer


@MODEL_INITIALIZER.register_module(module_name="Naive")
class NaiveMOELayer(BaseMoELayer):
    """naive MoE without ep parallel"""

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
        assert gpc.expert_parallel_size == 1, "do not support expert parallel"
        super().__init__(
            torch.nn.Linear(hidden_size, num_experts, bias=False),
            torch.nn.ModuleList(
                [
                    FeedForward(
                        hidden_size,
                        int(hidden_size * gpc.config.model.mlp_ratio),
                        out_features=hidden_size,
                        process_group=gpc.get_group(ParallelMode.TENSOR),
                        bias=False,
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(num_experts // ep_size)
                ]
            ),
            ep_group,
            ep_size,
            num_experts // ep_size,
        )

        self.topk = top_k

        assert gpc.expert_parallel_size == 1

    def forward(self, *inputs):
        x = inputs[0]
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x).softmax(dim=-1)

        expert_weights, expert_indices = torch.topk(scores, self.topk, dim=-1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        flat_expert_indices = expert_indices.view(-1)        

        x = x.repeat_interleave(self.topk, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts.wrapped_experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)

        if gpc.is_rank_for_log():
            print(y.view(*orig_shape), flush=True)

        return y.view(*orig_shape)
