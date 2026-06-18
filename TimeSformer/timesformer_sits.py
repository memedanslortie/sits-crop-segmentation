import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TimesformerModel

class TemporalAggregator(nn.Module):
    """
    Learned temporal aggregation via attention pooling.
    Replaces naive mean pooling to preserve temporal dynamics.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        # Learnable query that attends over T timesteps
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, padding_mask=None):
        """
        tokens: [B*H_grid*W_grid, T, C]
        padding_mask: [B*H_grid*W_grid, T] (True = ignore this timestep)
        """
        B_flat, T, C = tokens.shape
        query = self.query.expand(B_flat, -1, -1)  # [B*H*W, 1, C]
        
        # Calculate attention over time, ignoring masked/cloudy days
        aggregated, _ = self.attn(
            query, 
            tokens, 
            tokens, 
            key_padding_mask=padding_mask
        )
        return self.norm(aggregated.squeeze(1))  # [B*H*W, C]


class _UpBlock(nn.Module):
    """x2 bilinear upsample + double conv (a U-Net-style decoder stage)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class ProgressiveDecoder(nn.Module):
    """Progressive learned upsampling (8->16->32->64->128) instead of a single x16 bilinear
    jump, PLUS a high-resolution skip from the raw input image. The transformer provides the
    SEMANTICS (at 8x8), the input skip restores the fine spatial DETAIL the patchifier threw
    away — this is what sharpens parcel boundaries (cf. SETR-PUP / DPT)."""
    def __init__(self, hidden_dim, num_classes, in_channels, patch_size=16):
        super().__init__()
        n_up = patch_size.bit_length() - 1  # power-of-2 patch: 16 -> 4 upsamples (8->128)
        chans = [256, 128, 64, 32, 32, 32, 32][:n_up]
        self.proj = nn.Conv2d(hidden_dim, 256, 1)
        ups, prev = [], 256
        for c in chans:
            ups.append(_UpBlock(prev, c)); prev = c
        self.ups = nn.ModuleList(ups)
        # shallow stem on the (temporally-pooled) raw input -> full-res detail features
        self.input_stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(prev + 32, num_classes, 1)

    def forward(self, feats, input_hr):
        x = self.proj(feats)
        for up in self.ups:
            x = up(x)
        skip = self.input_stem(input_hr)
        return self.head(torch.cat([x, skip], dim=1))


class SITS_TimeSformer(nn.Module):
    def __init__(self, in_channels=10, num_classes=20, img_size=128, patch_size=16,
                 intermediate_layers=(4, 8, 12), high_res_decoder=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.intermediate_layers = intermediate_layers
        self.high_res_decoder = high_res_decoder
        
        print(" Importing pre-trained TimeSformer (Kinetics-400) with Hidden States Extraction...")
        # 1. Load the pre-trained backbone
        self.backbone = TimesformerModel.from_pretrained(
            "facebook/timesformer-base-finetuned-k400", 
            ignore_mismatched_sizes=True,
            image_size=img_size,
            use_safetensors=True
        )

        # 2. Weight Inflation (3 RGB -> 10 Multispectral)
        print("🛠️ Applying Weight Inflation (3 RGB -> 10 Multispectral)...")
        old_patch_embed = self.backbone.embeddings.patch_embeddings.projection
        new_patch_embed = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=old_patch_embed.out_channels, 
            kernel_size=old_patch_embed.kernel_size, 
            stride=old_patch_embed.stride, 
            padding=old_patch_embed.padding
        )
        
        old_weights = old_patch_embed.weight.data
        mean_weights = old_weights.mean(dim=1, keepdim=True)
        new_weights = mean_weights.repeat(1, in_channels, 1, 1) * (3.0 / in_channels)
        
        new_patch_embed.weight.data = new_weights
        if old_patch_embed.bias is not None:
            new_patch_embed.bias.data = old_patch_embed.bias.data
            
        self.backbone.embeddings.patch_embeddings.projection = new_patch_embed

        self.hidden_dim = self.backbone.config.hidden_size # 768

        # 3. Feature Fusion Layer (Claude's addition)
        # We extract 3 layers, each having 768 dims, so we project 768*3 back down to 768
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.intermediate_layers), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )

        # 4. Intelligent Temporal Aggregator (Our addition)
        self.temporal_aggregator = TemporalAggregator(embed_dim=self.hidden_dim)

        # 5. Spatial Decoder
        if high_res_decoder:
            print(" Using high-resolution progressive decoder (+ raw-input skip)")
            self.decoder = ProgressiveDecoder(self.hidden_dim, num_classes, in_channels, patch_size)
        else:
            self.decoder = nn.Sequential(
                nn.Conv2d(self.hidden_dim, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
                nn.Conv2d(256, num_classes, kernel_size=1)
            )

    def forward(self, x, padding_mask=None):
        B, T, C, H, W = x.shape

        # NOTE: HuggingFace's TimesformerModel.forward() does not accept an attention mask,
        # so padding/cloud masking cannot be injected into the backbone's space-time
        # attention (an earlier version passed `attention_mask=...` here — it was silently
        # ignored). We therefore feed the full sequence to the backbone and enforce masking
        # downstream in the TemporalAggregator, which is the layer that actually collapses
        # the temporal axis: padded and cloudy frames are excluded there via key_padding_mask.
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        
        # Extract layers (e.g., 4, 8, 12)
        selected = [outputs.hidden_states[i] for i in self.intermediate_layers]
        
        # Drop the CLS token (index 0) from every extracted layer
        selected = [layer_feats[:, 1:, :] for layer_feats in selected] # Shape: [B, T*H*W, 768]
        
        # Concatenate and fuse: [B, T*H*W, 768*3] -> [B, T*H*W, 768]
        fused = torch.cat(selected, dim=-1)
        fused = self.feature_fusion(fused)

        # Reshape to spatio-temporal grid: [B, T, grid_h, grid_w, C]
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        tokens = fused.view(B, T, grid_h, grid_w, self.hidden_dim)
        
        # Flatten spatially: [B * grid_h * grid_w, T, C]
        tokens_flat = tokens.permute(0, 2, 3, 1, 4).reshape(B * grid_h * grid_w, T, self.hidden_dim)
        
        # Prepare the mask
        mask_flat = None
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2).expand(B, grid_h, grid_w, T)
            mask_flat = mask_expanded.reshape(B * grid_h * grid_w, T)
            all_masked = mask_flat.all(dim=1, keepdim=True)
            mask_flat = mask_flat.masked_fill(all_masked, False)
            
        # Apply Learned Temporal Aggregation
        aggregated = self.temporal_aggregator(tokens_flat, padding_mask=mask_flat)
        
        # Reshape back to spatial feature map: [B, C, grid_h, grid_w]
        spatial_features = aggregated.view(B, grid_h, grid_w, self.hidden_dim).permute(0, 3, 1, 2)
        
        # Final Spatial Decoding
        if self.high_res_decoder:
            # Temporally-pooled raw input (masked mean over valid frames) -> high-res skip.
            if padding_mask is not None:
                valid = (~padding_mask).float()
                w = (valid / valid.sum(dim=1, keepdim=True).clamp(min=1.0)).to(x.dtype)
                input_hr = torch.einsum('bt,btchw->bchw', w, x)
            else:
                input_hr = x.mean(dim=1)
            logits = self.decoder(spatial_features, input_hr)
        else:
            logits = self.decoder(spatial_features)

        return logits