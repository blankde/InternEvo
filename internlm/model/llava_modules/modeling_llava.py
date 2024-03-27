# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.utils.checkpoint
# from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from internlm.model.modeling_internlm2 import PackedFlashLlama1D as InternLM2Model
from internlm.model.llava_modules.modeling_internvit import InternVisionModel

logger = logging.get_logger(__name__)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size, assuming square window

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, 'H and W must be divisible by window_size'

    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H * W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


class InternVLChatModel(nn.modules):
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionEncoderLayer', 'LlamaDecoderLayer', 'LlamaForCausalLM']

    def __init__(self, 
        force_image_size,
        select_layer,
        template,
        downsample_ratio,
        image_fold,
        ps_version,
        vision_config,
        llm_config,
        vision_model=None, 
        language_model=None
    ):
        super().__init__()

        image_size = force_image_size or vision_config.image_size
        patch_size = vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = select_layer
        self.template = template
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
        self.downsample_ratio = downsample_ratio
        self.image_fold = image_fold
        self.ps_version = ps_version

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = InternLM2Model(llm_config)

        vit_hidden_size = vision_config.hidden_size
        llm_hidden_size = llm_config.hidden_size

        #TODO support 3d parallel
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(int(vit_hidden_size / downsample_ratio ** 2)),
            nn.Linear(int(vit_hidden_size / downsample_ratio ** 2), llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        print(f"{force_image_size=}, {vision_config.image_size=}")
        if force_image_size != vision_config.image_size:
            self.vision_model.resize_pos_embeddings(
                old_size=vision_config.image_size,
                new_size=force_image_size,
                patch_size=vision_config.patch_size
            )
            self.vision_model.image_size = force_image_size

        self.img_context_token_id = None

        # if config.use_backbone_lora:
        #     self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        # if config.use_llm_lora:
        #     self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    # def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
    #     lora_config = LoraConfig(
    #         r=r,
    #         target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
    #         lora_alpha=lora_alpha,
    #         lora_dropout=lora_dropout,
    #     )
    #     self.vision_model = get_peft_model(self.vision_model, lora_config)
    #     self.vision_model.print_trainable_parameters()

    # def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
    #     lora_config = LoraConfig(
    #         r=r,
    #         target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    #                         'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'],
    #         lora_alpha=lora_alpha,
    #         lora_dropout=lora_dropout,
    #         task_type='CAUSAL_LM'
    #     )
    #     self.language_model = get_peft_model(self.language_model, lora_config)
    #     self.language_model.enable_input_require_grads()
    #     self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = min(selected.sum(), vit_embeds.shape[0])
            selected = selected[:n_token]
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.image_fold:
            image_size = pixel_values.size(-1)  # B, C, H, W
            pixel_values = window_partition(pixel_values, window_size=image_size // self.image_fold)  # 4B, C, H/2, W/2

        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        if self.image_fold:
            vit_embeds = window_reverse(vit_embeds, window_size=image_size // (self.image_fold * self.patch_size),
                                        H=image_size // self.patch_size, W=image_size // self.patch_size)

        # if torch.distributed.get_rank() == 0:
        #     print("before pixel shuffle:", vit_embeds.shape)
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        # if torch.distributed.get_rank() == 0:
        #     print("after pixel shuffle:", vit_embeds.shape)
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds
