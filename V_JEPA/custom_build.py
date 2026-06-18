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
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")

        checkpoint_key = _CHECKPOINT_KEY_MAP[model_name]
        encoder_state_dict = _clean_backbone_key(state_dict[checkpoint_key])

        # Load weights, ignoring size mismatches (pos_embed is handled by RoPE)
        encoder.load_state_dict(encoder_state_dict, strict=False)
        # log.info("Loaded pretrained weights (ignoring size mismatches).")
        if not is_rgb:
            # Replace the first layer of the patch embedding to accept `in_chans`
            # The patch_embed is a PatchEmbed3D module
            old_patch_embed = encoder.patch_embed
            new_patch_embed = nn.Conv3d(
                in_channels=3 if is_rgb else 10,
                out_channels=old_patch_embed.embed_dim,
                kernel_size=old_patch_embed.kernel_size,
                stride=old_patch_embed.stride,
                padding=old_patch_embed.padding,
                bias=old_patch_embed.bias is not None,
            )
            # Copy weights for the first 3 channels (if pretrained on 3 channels)
            with torch.no_grad():
                new_patch_embed.weight[:, :3] = old_patch_embed.weight
                if old_patch_embed.bias is not None:
                    new_patch_embed.bias.copy_(old_patch_embed.bias)
                # Initialize remaining channels randomly
                nn.init.kaiming_normal_(new_patch_embed.weight[:, 3:], mode="fan_out")

            # Replace the patch_embed
            encoder.patch_embed = new_patch_embed

    return encoder