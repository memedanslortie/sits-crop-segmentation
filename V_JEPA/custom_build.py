import os

import torch
import torch.nn as nn
from vjepa2.src.hub.backbones import _make_vjepa2_1_model, _clean_backbone_key, ARCH_NAME_MAP

from utils.logging import get_logger
log = get_logger()

_CHECKPOINT_KEY_MAP = {
    "vjepa2_1_vit_base_384":     "ema_encoder",
    "vjepa2_1_vit_large_384":    "ema_encoder",
    "vjepa2_1_vit_giant_384":    "target_encoder",
    "vjepa2_1_vit_gigantic_384": "target_encoder",
}

def build_vjepa2_1_encoder(
    model_name: str = "vjepa2_1_vit_large_384",
    is_rgb: bool = False,
    pretrained: bool = True,
    **kwargs,
):
    encoder, predictor = _make_vjepa2_1_model(
        model_name=model_name,
        img_size=128,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        pretrained=False,
        **kwargs,
    )
    del predictor

    if pretrained:
        VJEPA_BASE_URL = os.getenv("VJEPA_BASE_URL", "https://dl.fbaipublicfiles.com/vjepa2")
        model_file = ARCH_NAME_MAP[model_name][-1]
        url = VJEPA_BASE_URL + f"/{model_file}.pt"
        log.info("Start loading pretrained")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=False)
        checkpoint_key = _CHECKPOINT_KEY_MAP[model_name]
        encoder_state_dict = _clean_backbone_key(state_dict[checkpoint_key])
        encoder.load_state_dict(encoder_state_dict, strict=False)
        log.info("Loaded pretrained weights.")

    if not is_rgb:
        log.info("Adapting 1st layer for 10 bands")
        old_proj = encoder.patch_embed.proj
        new_proj = nn.Conv3d(
            in_channels=10,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            nn.init.kaiming_normal_(new_proj.weight, mode="fan_in")
            if pretrained:
                new_proj.weight[:, :3].copy_(old_proj.weight)
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        encoder.patch_embed.proj = new_proj

    return encoder