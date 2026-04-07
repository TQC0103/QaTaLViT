from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .improved_ssl import (
    BottleneckCrossAttention,
    CrossModalSkipAdapter,
    PrototypeMemoryBank,
    StructuredTextConditioner,
    TextSpatialPrior,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpsampleFusionBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, context_dim: int) -> None:
        super().__init__()
        self.skip_adapter = CrossModalSkipAdapter(skip_channels, context_dim)
        self.fuse = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        context: torch.Tensor,
        spatial_prior: torch.Tensor,
    ) -> torch.Tensor:
        upsampled = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        adapted_skip = self.skip_adapter(skip, context, spatial_prior)
        return self.fuse(torch.cat([upsampled, adapted_skip], dim=1))


class LViTImproved(nn.Module):
    """
    Improved LViT variant guided by the research report:
    - true bottleneck image-text cross-attention
    - explicit text-view spatial prior
    - prototype fallback when text is missing
    - cross-modal skip refinement during decoding
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        img_size: int = 224,
        text_dim: int = 768,
        base_channels: int = 64,
        bottleneck_dim: int = 512,
        num_bottleneck_tokens: int = 4,
        num_heads: int = 8,
        transformer_layers: int = 2,
        num_prototypes: int = 8,
        prototype_momentum: float = 0.1,
        attribute_dim: int = 0,
        text_dropout_prob: float = 0.3,
        enable_upper_scale_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.text_dim = text_dim
        self.bottleneck_dim = bottleneck_dim
        self.spatial_size = img_size // 16
        self.attribute_dim = attribute_dim
        self.enable_upper_scale_fusion = enable_upper_scale_fusion

        self.stem = ConvBlock(n_channels, base_channels)
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)
        self.down3 = DownsampleBlock(base_channels * 4, bottleneck_dim)
        self.down4 = DownsampleBlock(bottleneck_dim, bottleneck_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=num_heads,
            dim_feedforward=bottleneck_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.visual_bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.upper_scale_encoder = (
            nn.TransformerEncoder(encoder_layer, num_layers=1) if enable_upper_scale_fusion else None
        )
        self.text_conditioner = StructuredTextConditioner(
            text_dim=text_dim,
            model_dim=bottleneck_dim,
            attribute_dim=attribute_dim,
            text_dropout_prob=text_dropout_prob,
        )
        self.prototype_memory = PrototypeMemoryBank(
            dim=bottleneck_dim,
            num_prototypes=num_prototypes,
            momentum=prototype_momentum,
        )
        self.cross_attention = BottleneckCrossAttention(
            dim=bottleneck_dim,
            num_heads=num_heads,
            num_bottleneck_tokens=num_bottleneck_tokens,
        )
        self.upper_scale_cross_attention = (
            BottleneckCrossAttention(
                dim=bottleneck_dim,
                num_heads=num_heads,
                num_bottleneck_tokens=num_bottleneck_tokens,
            )
            if enable_upper_scale_fusion
            else None
        )
        self.text_prior = TextSpatialPrior(
            dim=bottleneck_dim,
            height=self.spatial_size,
            width=self.spatial_size,
            attribute_dim=attribute_dim,
        )
        self.prior_to_logits = nn.Conv2d(1, n_classes, kernel_size=1)

        self.up4 = UpsampleFusionBlock(bottleneck_dim, bottleneck_dim, base_channels * 4, bottleneck_dim)
        self.up3 = UpsampleFusionBlock(base_channels * 4, base_channels * 4, base_channels * 2, bottleneck_dim)
        self.up2 = UpsampleFusionBlock(base_channels * 2, base_channels * 2, base_channels, bottleneck_dim)
        self.up1 = UpsampleFusionBlock(base_channels, base_channels, base_channels, bottleneck_dim)

        self.segmentation_head = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def _prepare_text(
        self,
        text_tokens: Optional[torch.Tensor],
        structured_attributes: Optional[torch.Tensor],
        visual_summary: torch.Tensor,
        force_text_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        retrieved_summary, prototype_weights = self.prototype_memory.retrieve(visual_summary)
        prepared_tokens, text_summary, attribute_context, fallback_mask = self.text_conditioner(
            text_tokens=text_tokens,
            structured_attributes=structured_attributes,
            prototype_summary=retrieved_summary,
            force_drop_mask=force_text_dropout_mask,
        )
        return prepared_tokens, text_summary, retrieved_summary, prototype_weights, fallback_mask, attribute_context

    def forward(
        self,
        image: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        structured_attributes: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        update_prototypes: bool = False,
        force_text_dropout_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x1 = self.stem(image.float())
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        visual_tokens = x5.flatten(2).transpose(1, 2)
        visual_tokens = self.visual_bottleneck(visual_tokens)
        visual_summary = visual_tokens.mean(dim=1)

        text_tokens_prepared, text_summary, prototype_summary, prototype_weights, used_prototype_fallback, attribute_context = (
            self._prepare_text(
                text_tokens,
                structured_attributes,
                visual_summary,
                force_text_dropout_mask=force_text_dropout_mask,
            )
        )
        fused_tokens, bottleneck_tokens, gate = self.cross_attention(visual_tokens, text_tokens_prepared)
        fused_summary = fused_tokens.mean(dim=1)

        upper_gate = None
        if self.enable_upper_scale_fusion and self.upper_scale_cross_attention is not None and self.upper_scale_encoder is not None:
            upper_tokens = x4.flatten(2).transpose(1, 2)
            upper_tokens = self.upper_scale_encoder(upper_tokens)
            upper_tokens, _, upper_gate = self.upper_scale_cross_attention(upper_tokens, text_tokens_prepared)
            x4 = upper_tokens.transpose(1, 2).reshape_as(x4)

        if self.training and update_prototypes and text_tokens is not None:
            paired_mask = ~used_prototype_fallback
            self.prototype_memory.update(fused_summary.detach(), valid_mask=paired_mask)

        spatial_prior = self.text_prior(
            text_tokens=text_tokens_prepared if text_tokens is not None else None,
            fallback_summary=prototype_summary,
            structured_attributes=structured_attributes,
        )
        bottleneck_map = fused_tokens.transpose(1, 2).reshape(
            x5.shape[0],
            self.bottleneck_dim,
            self.spatial_size,
            self.spatial_size,
        )

        y4 = self.up4(bottleneck_map, x4, fused_summary, spatial_prior)
        y3 = self.up3(y4, x3, fused_summary, spatial_prior)
        y2 = self.up2(y3, x2, fused_summary, spatial_prior)
        y1 = self.up1(y2, x1, fused_summary, spatial_prior)

        logits = self.segmentation_head(y1)
        prior_logits = F.interpolate(
            self.prior_to_logits(spatial_prior),
            size=logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        combined_logits = logits + prior_logits + self.output_bias
        prior_prediction = torch.sigmoid(prior_logits)
        probabilities = torch.sigmoid(combined_logits)

        if not return_aux:
            return combined_logits

        aux = {
            "spatial_prior": spatial_prior,
            "prior_logits": prior_logits,
            "prior_prediction": prior_prediction,
            "probabilities": probabilities,
            "bottleneck_tokens": bottleneck_tokens,
            "fusion_gate": gate,
            "upper_scale_gate": upper_gate,
            "visual_summary": visual_summary,
            "fused_summary": fused_summary,
            "text_summary": text_summary,
            "prototype_summary": prototype_summary,
            "prototype_weights": prototype_weights,
            "attribute_context": attribute_context,
            "used_prototype_fallback": used_prototype_fallback.to(device=combined_logits.device, dtype=combined_logits.dtype),
        }
        return combined_logits, aux


def build_lvit_improved(**kwargs) -> LViTImproved:
    return LViTImproved(**kwargs)
