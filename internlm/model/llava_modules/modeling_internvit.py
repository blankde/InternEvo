from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from internlm.model.ops.linear import get_linear_cls
from internlm.core.context import global_context as gpc
from internlm.model.modules.multi_head_attention import SelfAttention, DistributedAttention
from internlm.model.modules.vit_mlp import get_vit_mlp_cls
from internlm.core.context import ParallelMode
from internlm.model.moe import MoE
from internlm.core.naive_amp import set_fp32_attr_to_module
from internlm.solver.activation_checkpoint import activation_checkpoint
from internlm.solver.pipeline_utils import partition_uniform
from internlm.utils.common import filter_kwargs
from internlm.utils.logger import get_logger
from internlm.utils.registry import MODEL_INITIALIZER
from internlm.model.utils import (
    gather_forward_split_backward,
    split_forward_gather_backward,
    try_import_RMSNorm,
)


MODEL_TYPE = "INTERNVIT"

logger = get_logger(__file__)
RMSNorm = try_import_RMSNorm()


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


#TODO need to support tensor parallel and sequence parallel
class VisionEmbeddings(nn.Module):
    def __init__(self, 
        embed_dim: int = 768,  
        image_size: int = 768,
        patch_size: int = 16,   
        dtype: torch.dtype = None,  
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim, dtype=dtype),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, dtype=dtype
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim, dtype=dtype), )

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
        embed_dim: int,
        num_heads: int,
        process_group: Optional[torch.distributed.ProcessGroup],
        sequence_process_group: Optional[torch.distributed.ProcessGroup],
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_normalization: bool = False,
        layer_norm_epsilon: float = 0.0,
        qkv_bias: bool = False,
        layer_idx: int = None,
        use_flash_attn: bool = True,
        norm_type: str = "rmsnorm",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tp_mode: str = "mtp",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_flash_attn = use_flash_attn
        self.head_dim = self.embed_dim // self.num_heads
        self.tp_mode = tp_mode
        self.layer_idx = layer_idx
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        # TODO embedding not support tp
        Wqkv_cls = get_linear_cls(self.tp_mode, "column")
        self.Wqkv = Wqkv_cls(
            embed_dim,
            3 * embed_dim,
            process_group,
            bias=qkv_bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.qk_normalization = qk_normalization

        if self.qk_normalization:
            if norm_type == "rmsnorm":
                self.q_norm = RMSNorm(self.embed_dim, eps=layer_norm_epsilon)
                self.k_norm = RMSNorm(self.embed_dim, eps=layer_norm_epsilon)
            else:
                self.q_norm = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon)
                self.k_norm = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon)

        if gpc.config.model.use_flash_attn:
            from flash_attn.modules.mha import FlashSelfAttention

            inner_attn_cls = FlashSelfAttention
        else:
            inner_attn_cls = SelfAttention
        self.inner_attn = inner_attn_cls(attention_dropout=attention_dropout)

        if self.tp_mode == "isp":
            self.inner_attn = DistributedAttention(self.inner_attn, sequence_process_group=sequence_process_group)

        # output projection always have the bias (for now)
        out_proj_cls = get_linear_cls(self.tp_mode, "row")
        self.proj = out_proj_cls(
            embed_dim,
            embed_dim,
            process_group,
            bias=True,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        if kwargs.get("indexes", None) is not None:
            return self._packed_forward(x=x, inference_params=inference_params, **kwargs)
        else:
            return self._forward(x=x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    def _forward(self, x, seqlen=None, **kwargs):  # pylint: disable=W0613
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if seqlen=None.
                If seqlen is not None, x is (batch * seqlen, hidden_dim). This is so that when we
                split x during sequence parallel, we split the batch * seqlen dimension
                (in case batch is small).
        """
        qkv = self.Wqkv(x)
        if seqlen is None:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim)
        else:
            qkv = rearrange(qkv, "(b s) (three h d) -> b s three h d", s=seqlen, three=3, d=self.head_dim)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        if gpc.config.model.dtype is torch.float32 and gpc.config.model.use_flash_attn:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if qkv.dtype not in [torch.float16, torch.bfloat16]:
                    qkv = qkv.to(torch.bfloat16)
                context = self.inner_attn(qkv).to(x.dtype)
        else:
            context = self.inner_attn(qkv)

        if seqlen is None:
            context = rearrange(context, "b s h d -> b s (h d)")
        else:
            context = rearrange(context, "b s h d -> (b s) (h d)")
        outs = self.proj(context)
        outs = self.proj_drop(outs)
        return outs


class InternVisionEncoderLayer(nn.Module):
    def __init__(self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        attn_attention_drop: float = 0,
        attn_proj_drop : float = 0,
        initializer_factor: float = 0,
        drop_path_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        checkpoint: bool = False,
        qk_normalization: bool = False,
        qkv_bias: bool = False,
        norm_type: str = "rmsnorm",
        layer_idx: int = 0,
        dtype: torch.dtype = torch.float,
        device: Optional[torch.device] = None,
        use_flash_attn: bool = True,
        tp_mode: str = "mtp",
        num_experts: int = 0,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.checkpoint = checkpoint
        self.layer_idx = layer_idx
        self.use_flash_attn = use_flash_attn
        self.tp_mode = tp_mode
        parallel_mode = ParallelMode.WEIGHT if self.tp_mode == "isp" else ParallelMode.TENSOR

        self.mixer = VisionAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            process_group=gpc.get_group(parallel_mode),
            sequence_process_group=gpc.get_group(ParallelMode.TENSOR),
            attention_dropout = attn_attention_drop,
            proj_dropout=attn_proj_drop,
            qk_normalization = qk_normalization,
            layer_norm_epsilon = layer_norm_epsilon,
            qkv_bias = qkv_bias,
            layer_idx=layer_idx,
            use_flash_attn=use_flash_attn,
            device=device,
            dtype=dtype,
            tp_mode=self.tp_mode,
        )

        self.num_experts = num_experts
        ep_size = gpc.get_world_size(ParallelMode.EXPERT)
        mlp_cls = get_vit_mlp_cls(self.tp_mode)
        if num_experts <= 1:  # dense, not MoE
            self.mlp = mlp_cls(
                hidden_size,
                int(hidden_size * mlp_ratio),
                out_features=hidden_size,
                process_group=gpc.get_group(parallel_mode),
                bias=False,
                device=device,
                dtype=dtype,
            )
        else:
            # replace mlp by MoE module. The expert in MoE is a FeedForward module.
            self.mlp = MoE(
                hidden_size=hidden_size,
                num_experts=num_experts,
                ep_group=gpc.get_group(ParallelMode.EXPERT),
                ep_size=ep_size,
                expert_cls=mlp_cls,
                device=device,
                dtype=dtype,
            )
            set_fp32_attr_to_module(self.mlp.moe_layer.gate)

        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
            self.norm2 = RMSNorm(hidden_size, eps=layer_norm_epsilon)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)

        self.ls1 = nn.Parameter(initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, hidden_states):
        if self.checkpoint and self.training:
            return activation_checkpoint(
                self._forward, False, hidden_states)
        else:
            return self._forward(hidden_states)

    def _forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self,
        num_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        attn_proj_drop: float = 0.0,
        initializer_factor: float = 0,
        drop_path_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-5,
        qk_normalization: bool = False,
        qkv_bias: bool = False,
        checkpoint: float = 0.0,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        norm_type: str = "rmsnorm",
        use_flash_attn: bool = True,
        output_hidden_states: bool = False,
        use_return_dict: bool = False,
        num_experts: int = 0,
    ):
        super().__init__()
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.use_flash_attn =use_flash_attn
        # stochastic depth decay rule
        if isinstance(gpc.config.parallel["tensor"], dict):
            self.tp_mode = gpc.config.parallel["tensor"].get("mode", "mtp")
        checkpoint_layer_num = int(num_layers * checkpoint)
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                mlp_ratio=mlp_ratio,
                attn_drop_rate=attn_drop_rate,
                attn_proj_drop=attn_proj_drop,
                initializer_factor=initializer_factor,
                drop_path_rate=drop_path_rate,
                layer_norm_epsilon=layer_norm_epsilon,
                qk_normalization= qk_normalization,
                qkv_bias= qkv_bias,
                norm_type=norm_type,
                layer_idx=lid + start_layer_idx,  # This parameter is used for caching during generation
                checkpoint=lid < checkpoint_layer_num,
                dtype=dtype,
                device=device,
                use_flash_attn=use_flash_attn,
                tp_mode=self.tp_mode,
                num_experts=num_experts,
                )
                for lid in range(num_layers)
            ]
        )

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            # return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds


        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(
                hidden_states,
            )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )


class InternVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self,
        num_layers: int = 12,
        hidden_size: int = 768,
        image_size: int = 768,
        patch_size: int = 16,  
        num_attention_heads: int = 12,
        mlp_ratio: int = 4.0,
        attn_drop_rate: float = 0.0,
        attn_proj_drop: float = 0.0,
        initializer_factor: float = 0,
        drop_path_rate: float = 0.0,
        dtype: torch.dtype = torch.float,
        layer_norm_epsilon: float = 1e-5,
        qk_normalization: bool = False,
        qkv_bias: bool = False,
        # first: bool = False, #TODO not support pipeline for now
        checkpoint: float = 0.0,
        start_layer_idx: int = 0,
        device: Optional[torch.device] = None,
        norm_type: str = "rmsnorm",
        use_flash_attn: bool = True,
        num_experts: int = 0,
    ):

        super().__init__()

        self.embeddings = VisionEmbeddings(
            embed_dim=hidden_size,  
            image_size=image_size,
            patch_size=patch_size,   
            dtype=dtype,
        )

        self.encoder = InternVisionEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            attn_drop_rate=attn_drop_rate,
            attn_proj_drop=attn_proj_drop,
            initializer_factor=initializer_factor,
            drop_path_rate=drop_path_rate,
            dtype=dtype,
            layer_norm_epsilon=layer_norm_epsilon,
            qk_normalization=qk_normalization,
            qkv_bias=qkv_bias,
            checkpoint=checkpoint,
            start_layer_idx=start_layer_idx,
            device=device,
            norm_type=norm_type,
            use_flash_attn=use_flash_attn,
            num_experts=num_experts,
        )


    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, _, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        if gpc.config.parallel.sequence_parallel and self.tp_mode == "isp":
            indexes = split_forward_gather_backward(indexes, ParallelMode.TENSOR, dim=0)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    

def build_intern_vision_model(vision_tower_cfg, device):
    """
    build generic model 1d

    Args:
        vision_tower_cfg: The intern vision model config.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    model = InternVisionModel(**filter_kwargs(InternVisionModel.__init__, vision_tower_cfg)).to(device)

    return model