import os

from .clip_encoder import CLIPVisionTower
from .modeling_internvit import build_intern_vision_model


def build_vision_tower(vision_tower_cfg=None, dtype=None, device=None,**kwargs):
    # print(vision_tower_cfg)
    vision_tower = vision_tower_cfg.get("mm_vision_tower", None)
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        model = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return model

    raise ValueError(f"Unknown vision tower: {vision_tower}")
