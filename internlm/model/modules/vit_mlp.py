#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from typing import Callable, Optional

import torch
from torch import nn


from transformers.activations import ACT2FN

from internlm.core.context import global_context as gpc
from internlm.model.ops.linear import (
    ColumnParallelLinearTorch,
    ISPLinear,
    MegatronColumnParallelLinearTorch,
    MegatronRowParallelLinearTorch,
    RowParallelLinearTorch,
)
from internlm.model.utils import Silu
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER


class BaseVisionFeedForward(nn.Module):
    """
    Base FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
        column_cls (Optional[Callable]): The column parallel class for w1 and w3. None by default.
        row_cls (Optional[Callable]): The row parallel class for w2. None by default.
    """


    def __init__(self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        # hidden_act = None, #TODO
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        column_cls: Optional[Callable] = None,
        row_cls: Optional[Callable] = None,
    ):
        super().__init__()
        self.act = ACT2FN["gelu"]
        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)


        self.fc1 = column_cls(
            in_features,
            hidden_features,
            process_group=process_group,
            bias=bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )
        self.fc2 = row_cls(
            hidden_features,
            out_features,
            process_group=process_group,
            bias=bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            device=device,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class VisionFeedForward(BaseVisionFeedForward):
    """
    FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            ColumnParallelLinearTorch,
            RowParallelLinearTorch,
        )


class MegatronVisionFeedForward(BaseVisionFeedForward):
    """
    FeedForward in megatron implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            MegatronColumnParallelLinearTorch,
            MegatronRowParallelLinearTorch,
        )


class ISPVisionFeedForward(BaseVisionFeedForward):
    """
    FeedForward in ISP.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        process_group (Optional[torch.distributed.ProcessGroup]): The group of the current device for `parallel_mode`.
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
    ):
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            process_group,
            bias,
            device,
            dtype,
            multiple_of,
            ISPLinear,
            ISPLinear,
        )


def get_vit_mlp_cls(tp_mode: str):
    if tp_mode in ["mtp", "fsp"]:
        mlp_cls = VisionFeedForward
    elif tp_mode == "msp":
        mlp_cls = MegatronVisionFeedForward
    else:
        mlp_cls = ISPVisionFeedForward
    return mlp_cls
