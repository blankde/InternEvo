#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from internlm.core.context import IS_REPLICA_ZERO_PARALLEL, ParallelMode
from internlm.core.context.parallel_context import global_context as gpc
from internlm.initialize.initialize_tensor import normal_, scaled_init_method_normal
from internlm.model.base_model import BaseModel
from internlm.model.modules.embedding import Embedding1D
from internlm.model.modules.linear import new_linear
from internlm.model.modules.mha import MHA
from internlm.model.modules.mlp import new_feed_forward
from internlm.model.modules.norm import new_layer_norm
from internlm.model.moe.moe import MoE
from internlm.model.utils import (
    convert_attn_args_to_kwargs,
    convert_attn_kwargs_to_args,
    internlm1_mha_pre_load_convert,
    internlm1_mha_save_convert,
)
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.utils.logger import get_logger

logger = get_logger(__file__)


class MegatronMLP(nn.Module):
    """
    Base FeedForward in flash implementation.

    Args:
        in_features (int): size of each input sample
        hidden_features (int): size of hidden state of FFN
        out_features (int): size of each output sample
        bias (bool): Whether the bias is needed for linears. True by default. But it is typically set to False
                    in the config.
        device (Optional[Union[str, torch.device]]): The device will be used.
        dtype (Optional[torch.dtype]): The type of data.
        multiple_of (int): For efficient training. Reset the size of hidden feature. 256 by default.
        mlp_layer_fusion (Optional[Bool]):  Some linears without bias in FFN can be fused to reduce the comm cost of SP.
        activation_type (str): the activation function used for feed forward, "swiglu" by default.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = None,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        multiple_of: int = 256,
        mlp_layer_fusion: Optional[bool] = False,
        activation_type: str = "swiglu",
        is_expert: bool = False,
    ):
        super().__init__()

        # TODO: support gelu...
        assert activation_type in ("swiglu"), f"Unsupported activation type: {activation_type}"

        self.mlp_layer_fusion = mlp_layer_fusion

        hidden_features = multiple_of * ((hidden_features + multiple_of - 1) // multiple_of)

        self.w1 = new_linear("w1", in_features, hidden_features, bias, device=device, dtype=dtype, is_expert=is_expert)
        self.w2 = new_linear("w2", hidden_features, out_features, bias, device=device, dtype=dtype, is_expert=is_expert)

    def forward(self, x):
        w1_o = self.w1(x)
        out = self.w2(F.gelu(w1_o))
        return out


class GPTMoEDecoder(nn.Module):
    """
    InternLM1 MoE Decoder Layer.

    Args:
        hidden_size (int): The hidden size of model. 768 by default.
        num_attention_heads (int): The number of attention heads. 12 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0 by default.
        drop_rate (float): The dropout rate of the input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): Type of data. torch.float by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        checkpoint (bool): Whether to use checkpointing to save VRAM. True by default.
        layer_idx (int): The index of current layer. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used.
        norm_type (str): Use RMS norm or layernorm."rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout layers only.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        rope_base (int): The value of `base` for rotary position embeddings. 10000 by default.
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_drop_rate: float = 0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        num_experts: int = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_layer_kwargs: dict = None,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        # dropout selective checkpoint can only be enabled when checkpoint is disabled.
        self.dropout_selective_checkpoint = dropout_selective_checkpoint is True and checkpoint is False
        self.layer_idx = layer_idx

        head_dim = hidden_size // num_attention_heads

        self.mixer = MHA(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attn_drop_rate,
            max_position_embeddings=max_position_embeddings,
            softmax_scale=1 / math.sqrt(head_dim),
            causal=True,
            layer_idx=layer_idx,
            use_dynamic_ntk_rope=use_dynamic_ntk_rope,
            rotary_emb_dim=0,
            rotary_emb_scale_base=0,
            device=device,
            dtype=dtype,
            qk_interleaved=qk_interleaved,
            bias=False,
            out_bias=False,
        )

        # Compatible with the name of internlm1 Wqkv linear layer
        self.mixer.register_checkpoint_compatibility_hooks(internlm1_mha_pre_load_convert, internlm1_mha_save_convert)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        self.norm1 = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
        self.norm2 = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)

        self.num_experts = num_experts
        if num_experts <= 1:  # dense, not MoE
            self.mlp = new_feed_forward(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                bias=False,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                # TODO: to support more activation functions
                activation_type="swiglu" if use_swiglu else "swiglu",
            )
        else:
            # replace mlp by MoE module. The expert in MoE is a FeedForward module.
            # mlp_cls = get_mlp_cls(self.tp_mode)
            self.mlp = MoE(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
                moe_layer_kwargs=moe_layer_kwargs,
                device=device,
                dtype=dtype,
                mlp_layer_fusion=mlp_layer_fusion,
                multiple_of=multiple_of,
                # TODO: to support more activation functions
                activation_type="swiglu" if use_swiglu else "swiglu",
                custom_ffn=MegatronMLP,
            )

        self.use_swiglu = use_swiglu
        self.use_scaled_init = use_scaled_init
        self.residual_in_fp32 = residual_in_fp32  # only make sense when using prenorm
        self.return_residual = False
        self.reset_parameters()  # TODO: check this should be changed when moe is added

    def reset_parameters(self):
        with torch.no_grad():
            for name, param in self.mixer.named_parameters():
                if param.ndim == 1:
                    param.data.zero_()
                elif "wqkv" in name:
                    normal_(std=0.006)(param.data)
                elif self.use_scaled_init:
                    scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                else:
                    normal_(std=0.0015)(param.data)

            for name, param in self.mlp.named_parameters():
                if param.ndim == 1 and "bias" in name:
                    param.data.zero_()
                elif self.use_swiglu:
                    if self.use_scaled_init and "w2" in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        # candidate: w1, w3, fused_w1_w3
                        normal_(std=0.006 if "w1" in name or "w3" in name else 0.0015)(param.data)
                else:
                    if self.use_scaled_init and "fc1" not in name:
                        scaled_init_method_normal(sigma=0.006, num_layers=self.layer_idx + 1)(param.data)
                    else:
                        normal_(std=0.006 if "fc1" in name else 0.0015)(param.data)

    def forward(self, hidden_states, **kwargs):
        if self.checkpoint and self.training:
            # TODO: check whether this will be affected by moe
            # NOTICE: activation_checkpiont do not support kwargs when use_reentrant = True.
            args = convert_attn_kwargs_to_args(kwargs)
            return activation_checkpoint(self._forward, False, hidden_states, *args)
        else:
            return self._forward(hidden_states, **kwargs)

    def _forward(self, hidden_states, *args, **kwargs):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Attn/MLP(LN(residual))
            cu_seqlens: 1d LongTensor, len(cu_seqlens) = hidden_states + 1
            indexes: the length of index is same as hidden states, which stand for the current position
        """

        def _dropout_and_norm_attn(_hidden_states):
            _dropped = self.dropout1(_hidden_states)
            _residual = _dropped
            _hidden_states = self.norm1(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_attn, False, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_attn(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        mixer_kwargs = convert_attn_args_to_kwargs(args, kwargs)
        hidden_states = self.mixer(hidden_states, **mixer_kwargs)

        def _dropout_and_norm_ffn(_residual, _hidden_states):
            _dropped = self.dropout2(_hidden_states)
            _residual = (_dropped + _residual) if _residual is not None else _dropped
            _hidden_states = self.norm2(_residual.float())
            return _residual, _hidden_states

        if self.dropout_selective_checkpoint:
            residual, hidden_states = activation_checkpoint(_dropout_and_norm_ffn, False, residual, hidden_states)
        else:
            residual, hidden_states = _dropout_and_norm_ffn(residual, hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # MLP.
        if self.num_experts <= 1:  # dense mlp output
            hidden_states = self.mlp(hidden_states)
            moe_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        else:  # MoE output
            hidden_states, moe_loss, _ = self.mlp(hidden_states)

        return hidden_states + residual, moe_loss


class GPTMoE(BaseModel):
    """
    InternLM1 MoE.

    Args:
        num_layers (int): The number of layer. 12 by default.
        hidden_size (int): The size of hidden state. 768 by default.
        num_attention_heads (int): The number of attention head. 12 by default.
        vocab_size (int): The size of vocabulary. 50304 by default.
        mlp_ratio (int): The ratio of MLP layers. 4 by default.
        attn_drop_rate (float): The dropout rate of attention module. 0.0 by default.
        drop_rate (float): The dropout rate of input hidden state. 0.0 by default.
        max_position_embeddings (int): The maximum position embeddings. 2048 by default.
        dtype (torch.dtype): The type of data. torch.float by default.
        checkpoint (float): The proportion of layers that need to be checkpointed compared to the total number
                                    of layers. 0.0 by default.
        layer_norm_epsilon (float): A value added to the denominator for numerical stability. 1e-5 by default.
        first (bool): Whether input embedding layer or not. False by default.
        last (bool): Whether output embedding layer or not. False by default.
        embed_grad_scale (float): Refer to GLM-130B, for training stability. 0.1 by default.
        parallel_output (bool): If it is necessary to collect the output of parallel computing. True by default.
        start_layer_idx (int): The index of start layer in the pipeline. 0 by default.
        use_dynamic_ntk_rope (bool): Whether to use dynamic ntk rope. False by default.
        device (Optional[Union[str, torch.device]]): The device will be used. None by default.
        residual_in_fp32 (bool): Whether to use residual in fp32. False by default.
        norm_type (str): Normalization type. Use RMSNorm or LayerNorm. "rmsnorm" by default.
        qk_interleaved (bool): Whether the odd and even columns of the wq and wk are normally interleaved.
        dropout_selective_checkpoint (bool): Whether to selectively checkpoint dropout and norm layers.
        use_scaled_init (bool): Whether to use scaled initialization for weights.
        use_swiglu (bool): Whether to use SwiGLU activation in the mlp module.
        num_experts (int): The number of experts. <=1 means dense, >1 means MoE. 1 by default.
        moe_use_residual (bool, optional): default=False, make this MoE layer a Residual MoE
                                          (https://arxiv.org/abs/2201.05596) layer.
        moe_type (str): determine which moe impl will be used, default is GShardMoE
        mlp_layer_fusion (bool): Whether to fuse layers in the mlp module for optimization.
        multiple_of (int): Ensures mlp dimensions are multiples of this value for efficient hardware utilization.
    """

    def __init__(
        self,
        num_layers: int = 48,
        hidden_size: int = 2048,
        num_attention_heads: int = 32,
        vocab_size: int = 50304,
        mlp_ratio: float = 4.0,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        max_position_embeddings: int = 2048,
        dtype: torch.dtype = torch.float,
        checkpoint: float = 0.0,
        layer_norm_epsilon: float = 1e-5,
        first: bool = False,
        last: bool = False,
        embed_grad_scale: float = 0.1,
        parallel_output: bool = True,
        start_layer_idx: int = 0,
        use_dynamic_ntk_rope: bool = False,
        device: Optional[torch.device] = None,
        residual_in_fp32: bool = False,
        norm_type: str = "rmsnorm",
        qk_interleaved: bool = False,
        is_reward: bool = False,
        dropout_selective_checkpoint: bool = True,
        use_scaled_init: bool = True,
        use_swiglu: bool = True,
        mlp_layer_fusion: bool = False,
        multiple_of: int = 256,
        moe_type: str = None,  # pylint: disable=W0613
        num_experts: bool = 1,
        top_k: int = 1,
        num_shared_experts: int = 0,
        moe_layer_kwargs: dict = None,
        tie_embeddings_and_output_weights: bool = False,
    ):
        super().__init__()

        self.tie_embeddings_and_output_weights = tie_embeddings_and_output_weights

        checkpoint_layer_num = int(num_layers * checkpoint)

        if first:
            self.embedding = Embedding1D(
                num_embeddings=vocab_size, embedding_dim=hidden_size, vocab_parallel=True, device=device
            )

            self.position_embeddings = Embedding1D(
                num_embeddings=vocab_size, embedding_dim=hidden_size, vocab_parallel=True, device=device
            )

            for param in self.position_embeddings.parameters():
                setattr(param, IS_REPLICA_ZERO_PARALLEL, True)

            for _, param in self.embedding.named_parameters():
                normal_(std=0.0052)(param)
        self.embed_grad_scale = embed_grad_scale
        self.blocks = nn.ModuleList(
            [
                GPTMoEDecoder(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    max_position_embeddings=max_position_embeddings,
                    dtype=dtype,
                    layer_norm_epsilon=layer_norm_epsilon,
                    checkpoint=lid < checkpoint_layer_num,
                    layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                    use_dynamic_ntk_rope=use_dynamic_ntk_rope,
                    residual_in_fp32=residual_in_fp32,
                    device=device,
                    norm_type=norm_type,
                    dropout_selective_checkpoint=dropout_selective_checkpoint,
                    use_scaled_init=use_scaled_init,
                    use_swiglu=use_swiglu,
                    qk_interleaved=qk_interleaved,
                    mlp_layer_fusion=mlp_layer_fusion,
                    multiple_of=multiple_of,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_shared_experts=num_shared_experts,
                    moe_layer_kwargs=moe_layer_kwargs,
                )
                for lid in range(num_layers)
            ]
        )
        if last:
            self.norm = new_layer_norm(norm_type, hidden_size, eps=layer_norm_epsilon)
            self.head = new_linear(
                name="head",
                in_features=hidden_size,
                out_features=gpc.get_world_size(ParallelMode.TENSOR) if is_reward else vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
                is_reward=is_reward,
                weight_scale=embed_grad_scale,
                skip_weight_alloction=tie_embeddings_and_output_weights,
            )
            for _, param in self.head.named_parameters():
                normal_(std=0.0052)(param)

        self.parallel_output = parallel_output

        if self.tie_embeddings_and_output_weights:
            self.initialize_word_embeddings(hidden_size, vocab_size, device)

    def forward(self, hidden_states=None, input_ids=None, **kwargs):
        # attention_mask: compute attention on the places where the value is 1
        # old condition may fail when use shared embedding
        if gpc.is_pipeline_first_stage() and input_ids is not None:
            tok_embeddings = self.embedding(input_ids)
            pos_embeddings = self.position_embeddings(kwargs["indexes"].view(input_ids.shape))
            hidden_states = tok_embeddings + pos_embeddings
            if self.embed_grad_scale != 1:
                hidden_states = (
                    self.embed_grad_scale * hidden_states + (1 - self.embed_grad_scale) * hidden_states.detach()
                )

        moe_losses = []
        for _, block in enumerate(self.blocks):
            hidden_states, mos_loss = block(hidden_states, **kwargs)
            moe_losses.append(mos_loss)

        if hasattr(self, "norm"):
            hidden_states = self.norm(hidden_states.float())
        if hasattr(self, "head"):
            if self.tie_embeddings_and_output_weights:
                hidden_states = self.head(hidden_states, self.shared_embedding_weight())
            else:
                hidden_states = self.head(hidden_states)

        return hidden_states, moe_losses

    def shared_embedding_weight(self):
        if not self.tie_embeddings_and_output_weights:
            raise Exception(
                "shared_embedding_weight() called for last stage, but share_embeddings_and_output_weights is false"
            )

        return self.embedding.weight

    def initialize_word_embeddings(
        self,
        hidden_size: int = 768,
        vocab_size: int = 50304,
        device: Optional[torch.device] = None,
    ):
        if not self.tie_embeddings_and_output_weights:
            raise Exception("initialize_word_embeddings() was called but tie_embeddings_and_output_weights is false")

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. Nothing to do if we aren't
        # using pipeline parallelism.
        if gpc.get_world_size(ParallelMode.PIPELINE) == 1:
            return

        if gpc.is_pipeline_first_stage():
            self.shared_embedding_weight().shared_embedding = True

        # Parameters are shared between the word embeddings layers, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before step perform an all-reduce between the
        #    grads of the two word_embeddings layers to ensure that every applied
        #    weight update is the same on both stages.
        if gpc.is_pipeline_last_stage():
            assert not gpc.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.embedding = Embedding1D(
                num_embeddings=vocab_size, embedding_dim=hidden_size, vocab_parallel=True, device=device
            )
            self.shared_embedding_weight().data.fill_(0)
            self.shared_embedding_weight().shared_embedding = True

        # Ensure that first and last stages have the same initial parameter
        # values.
        if gpc.is_pipeline_first_stage() or gpc.is_pipeline_last_stage():
            torch.distributed.all_reduce(
                self.shared_embedding_weight().data, group=gpc.get_group(ParallelMode.EMBEDDING_HEAD)
            )

    @staticmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError

    @staticmethod
    def convert_internevo2hf_weights(src: str, tgt: str) -> None:
        raise NotImplementedError
