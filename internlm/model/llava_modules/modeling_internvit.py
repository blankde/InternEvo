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

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False
from .configuration_intern_vit import InternVisionConfig
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
        config: InternVisionConfig,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim, dtype=config.dtype, device=config.device),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, dtype=config.dtype, device=config.device
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim, dtype=config.dtype, device=config.device), )

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
        config: InternVisionConfig,
        tp_mode: str = "mtp",
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        factory_kwargs = {"device": config.device, "dtype": config.dtype}
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )
        self.tp_mode = tp_mode

        # TODO embedding not support tp
        Wqkv_cls = get_linear_cls(self.tp_mode, "column")
        self.Wqkv = Wqkv_cls(
            self.embed_dim,
            3 * self.embed_dim,
            process_group,
            bias=self.qkv_bias,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )
        self.attn_drop = nn.Dropout(self.attention_dropout)
        self.proj_drop = nn.Dropout(self.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)

        # output projection always have the bias (for now)
        out_proj_cls = get_linear_cls(self.tp_mode, "row")
        self.proj = out_proj_cls(
            self.embed_dim,
            self.embed_dim,
            process_group,
            bias=False,
            sequence_parallel=gpc.config.parallel.sequence_parallel,
            **factory_kwargs,
        )

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return x


class InternVisionEncoderLayer(nn.Module):
    def __init__(self,
        config: InternVisionConfig,
        drop_path_rate: float,
        gradient_checkpointing: float = 0.0,
        tp_mode: str = "mtp",
    ):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.mlp_ratio = config.mlp_ratio
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flash_attn = config.use_flash_attn
        self.tp_mode = tp_mode
        parallel_mode = ParallelMode.WEIGHT if self.tp_mode == "isp" else ParallelMode.TENSOR

        self.attn = VisionAttention(
            config,
            process_group=gpc.get_group(parallel_mode),
            tp_mode=self.tp_mode,
        )

        self.num_experts = config.num_experts
        ep_size = gpc.get_world_size(ParallelMode.EXPERT)
        mlp_cls = get_vit_mlp_cls(self.tp_mode)
        if self.num_experts <= 1:  # dense, not MoE
            self.mlp = mlp_cls(
                config.hidden_size,
                int(config.hidden_size * config.mlp_ratio),
                out_features=config.hidden_size,
                process_group=gpc.get_group(parallel_mode),
                bias=False,
                device=config.device,
                dtype=config.dtype,
            )
        else:
            # replace mlp by MoE module. The expert in MoE is a FeedForward module.
            self.mlp = MoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                ep_group=gpc.get_group(ParallelMode.EXPERT),
                ep_size=ep_size,
                expert_cls=mlp_cls,
                device=config.device,
                dtype=config.dtype,
            )
            set_fp32_attr_to_module(self.mlp.moe_layer.gate)

        self.norm1 = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, hidden_states):
        if self.gradient_checkpointing and self.training:
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
        config: InternVisionConfig,
    ):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # stochastic depth decay rule
        if isinstance(gpc.config.parallel["tensor"], dict):
            self.tp_mode = gpc.config.parallel["tensor"].get("mode", "mtp")
        checkpoint_layer_num = int(config.num_hidden_layers * config.gradient_checkpointing)
        self.layers = nn.ModuleList(
            [
                InternVisionEncoderLayer(
                    config,
                    drop_path_rate=dpr,
                    gradient_checkpointing=lid < checkpoint_layer_num,
                    tp_mode=self.tp_mode,
                )
                for lid in range(config.num_hidden_layers)
            ]
        )

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
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
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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


class InternVisionModel(nn.Module):
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self,
        config: InternVisionConfig,
    ):

        super().__init__()
        self.config = config
        self.embeddings = VisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)


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
    

def build_intern_vision_model(vision_tower_cfg, dtype=None, device=torch.device("cuda")):
    """
    build generic model 1d

    Args:
        vision_tower_cfg: The intern vision model config.
        device (Optional[Union[str, torch.device]]): The device will be used. torch.device("cuda") by default.

    """
    vision_tower_cfg["device"] = device
    vision_tower_cfg["dtype"] = dtype
    model = InternVisionModel(vision_tower_cfg).to(device)

    return model