import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckCrossAttention(nn.Module):
    """True image-text interaction at the semantic bottleneck."""

    def __init__(self, dim: int, num_heads: int = 8, num_bottleneck_tokens: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.num_bottleneck_tokens = num_bottleneck_tokens
        self.bottleneck_tokens = nn.Parameter(torch.randn(num_bottleneck_tokens, dim) * 0.02)
        self.text_to_bottleneck = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.vision_to_bottleneck = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.bottleneck_to_vision = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = visual_tokens.shape[0]
        bottleneck = self.bottleneck_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        if text_tokens is None:
            text_tokens = visual_tokens.new_zeros(batch_size, 1, self.dim)

        bottleneck = bottleneck + self.text_to_bottleneck(bottleneck, text_tokens, text_tokens)[0]
        bottleneck = bottleneck + self.vision_to_bottleneck(bottleneck, visual_tokens, visual_tokens)[0]

        pooled_visual = visual_tokens.mean(dim=1)
        pooled_text = text_tokens.mean(dim=1)
        gate = self.gate(torch.cat([pooled_visual, pooled_text], dim=-1)).unsqueeze(1)

        attended_visual = self.bottleneck_to_vision(visual_tokens, bottleneck, bottleneck)[0]
        fused_visual = self.output_norm(visual_tokens + gate * attended_visual)
        return fused_visual, bottleneck, gate


class StructuredTextConditioner(nn.Module):
    """Projects text tokens and structured attributes into the fusion space."""

    def __init__(
        self,
        text_dim: int,
        model_dim: int,
        attribute_dim: int = 0,
        text_dropout_prob: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.attribute_dim = attribute_dim
        self.text_dropout_prob = text_dropout_prob
        self.token_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, model_dim),
        )
        self.attribute_projection = (
            nn.Sequential(
                nn.LayerNorm(attribute_dim),
                nn.Linear(attribute_dim, model_dim),
                nn.GELU(),
            )
            if attribute_dim > 0
            else None
        )
        self.no_text_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

    def forward(
        self,
        text_tokens: Optional[torch.Tensor],
        structured_attributes: Optional[torch.Tensor],
        prototype_summary: torch.Tensor,
        force_drop_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = prototype_summary.shape[0]
        device = prototype_summary.device
        dtype = prototype_summary.dtype

        if self.attribute_projection is not None and structured_attributes is not None:
            structured_attributes = structured_attributes.to(device=device, dtype=dtype)
            attribute_context = self.attribute_projection(structured_attributes)
        else:
            attribute_context = prototype_summary.new_zeros(batch_size, self.model_dim)

        if text_tokens is not None:
            projected_text = self.token_projection(text_tokens.float())
            projected_text = projected_text.to(device=device)
            projected_text = projected_text + attribute_context.unsqueeze(1)
            fallback_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
            if self.training and self.text_dropout_prob > 0:
                fallback_mask = torch.rand(batch_size, device=device) < self.text_dropout_prob
            if force_drop_mask is not None:
                fallback_mask = fallback_mask | force_drop_mask.to(device=device, dtype=torch.bool)
        else:
            projected_text = None
            fallback_mask = torch.ones(batch_size, device=device, dtype=torch.bool)

        fallback_tokens = self.no_text_token.expand(batch_size, -1, -1) + attribute_context.unsqueeze(1)
        text_summary = prototype_summary + attribute_context

        if projected_text is None:
            fallback_tokens = fallback_tokens.to(device=device, dtype=prototype_summary.dtype)
            prepared_tokens = fallback_tokens
        else:
            fallback_tokens = fallback_tokens.to(device=projected_text.device, dtype=projected_text.dtype)
            prepared_tokens = projected_text.clone()
            expanded_fallback = fallback_tokens.expand(-1, projected_text.shape[1], -1)
            text_summary = projected_text.mean(dim=1).to(device=device, dtype=prototype_summary.dtype)
            prepared_tokens[fallback_mask] = expanded_fallback[fallback_mask]
            text_summary[fallback_mask] = prototype_summary[fallback_mask] + attribute_context[fallback_mask]

        return prepared_tokens, text_summary, attribute_context, fallback_mask


class TextSpatialPrior(nn.Module):
    """Converts a text summary into a coarse probabilistic spatial prior map."""

    def __init__(self, dim: int, height: int, width: int, num_basis: int = 7, attribute_dim: int = 0) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_basis = num_basis
        self.attribute_dim = attribute_dim
        self.register_buffer("basis_maps", self._build_basis_maps(height, width, num_basis), persistent=False)
        self.summary_to_basis = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_basis),
        )
        self.attribute_to_basis = nn.Linear(attribute_dim, num_basis) if attribute_dim > 0 else None
        self.refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    @staticmethod
    def _build_basis_maps(height: int, width: int, num_basis: int) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height)
        x = torch.linspace(-1.0, 1.0, width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        center = torch.exp(-(xx.square() + yy.square()) / 0.5)
        left = torch.clamp(-xx, min=0.0)
        right = torch.clamp(xx, min=0.0)
        upper = torch.clamp(-yy, min=0.0)
        lower = torch.clamp(yy, min=0.0)
        vertical_band = torch.exp(-xx.square() / 0.25)
        horizontal_band = torch.exp(-yy.square() / 0.25)

        basis = torch.stack(
            [center, left, right, upper, lower, vertical_band, horizontal_band],
            dim=0,
        )
        if num_basis < basis.shape[0]:
            basis = basis[:num_basis]
        elif num_basis > basis.shape[0]:
            repeats = math.ceil(num_basis / basis.shape[0])
            basis = basis.repeat(repeats, 1, 1)[:num_basis]
        basis = basis / basis.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return basis

    def forward(
        self,
        text_tokens: Optional[torch.Tensor] = None,
        fallback_summary: Optional[torch.Tensor] = None,
        structured_attributes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if text_tokens is None:
            if fallback_summary is None:
                batch_size = 1
                device = self.basis_maps.device
                dtype = self.basis_maps.dtype
                return torch.zeros(batch_size, 1, self.height, self.width, device=device, dtype=dtype)
            summary = fallback_summary
        else:
            summary = text_tokens.mean(dim=1)

        logits = self.summary_to_basis(summary)
        if self.attribute_to_basis is not None and structured_attributes is not None:
            logits = logits + self.attribute_to_basis(structured_attributes.to(device=summary.device, dtype=summary.dtype))
        weights = torch.softmax(logits, dim=-1)
        prior = torch.einsum("bk,khw->bhw", weights, self.basis_maps)
        prior = prior.unsqueeze(1)
        return torch.sigmoid(self.refine(prior) + prior)


class PrototypeMemoryBank(nn.Module):
    """Stores semantic prototypes so the model can fall back when text is missing."""

    def __init__(self, dim: int, num_prototypes: int = 8, momentum: float = 0.1, temperature: float = 0.1) -> None:
        super().__init__()
        memory = F.normalize(torch.randn(num_prototypes, dim), dim=-1)
        self.register_buffer("memory", memory)
        self.momentum = momentum
        self.temperature = temperature

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized_query = F.normalize(query, dim=-1)
        normalized_memory = F.normalize(self.memory, dim=-1)
        logits = normalized_query @ normalized_memory.transpose(0, 1) / self.temperature
        weights = torch.softmax(logits, dim=-1)
        retrieved = weights @ self.memory
        return retrieved, weights

    @torch.no_grad()
    def update(self, features: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> None:
        features = features.to(device=self.memory.device, dtype=self.memory.dtype)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=features.device, dtype=torch.bool)
            if not valid_mask.any():
                return
            features = features[valid_mask]
        normalized_features = F.normalize(features, dim=-1)
        _, weights = self.retrieve(normalized_features)
        assignment_mass = weights.sum(dim=0, keepdim=True).transpose(0, 1)
        aggregated = weights.transpose(0, 1) @ normalized_features
        valid = assignment_mass.squeeze(-1) > 0
        if valid.any():
            aggregated[valid] = aggregated[valid] / assignment_mass[valid].clamp_min(1e-6)
            updated = self.memory.clone()
            updated[valid] = F.normalize(
                (1.0 - self.momentum) * updated[valid] + self.momentum * aggregated[valid],
                dim=-1,
            )
            self.memory.copy_(updated)


class CrossModalSkipAdapter(nn.Module):
    """Modulates decoder skip features using fused bottleneck context and the text prior."""

    def __init__(self, skip_channels: int, context_dim: int) -> None:
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, skip_channels),
            nn.GELU(),
            nn.Linear(skip_channels, skip_channels),
        )
        self.spatial_gate = nn.Conv2d(1, skip_channels, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_channels),
            nn.GELU(),
        )

    def forward(self, skip: torch.Tensor, context: torch.Tensor, spatial_prior: torch.Tensor) -> torch.Tensor:
        channel = torch.sigmoid(self.channel_gate(context)).unsqueeze(-1).unsqueeze(-1)
        prior = F.interpolate(spatial_prior, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        spatial = torch.sigmoid(self.spatial_gate(prior))
        adapted = skip * (1.0 + channel) * (1.0 + spatial)
        return self.refine(adapted)


class UncertaintyAwarePseudoLabelFusion(nn.Module):
    """Merges visual pseudo-labels and text priors only when they are reliable and agreeing."""

    def __init__(
        self,
        entropy_threshold: float = 0.35,
        agreement_threshold: float = 0.5,
        coarse_size: int = 16,
    ) -> None:
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.agreement_threshold = agreement_threshold
        self.coarse_size = coarse_size

    @staticmethod
    def _binary_entropy(probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        return -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())

    @staticmethod
    def _dice_score(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        lhs = lhs.reshape(lhs.shape[0], -1)
        rhs = rhs.reshape(rhs.shape[0], -1)
        intersection = (lhs * rhs).sum(dim=1)
        union = lhs.sum(dim=1) + rhs.sum(dim=1)
        return (2.0 * intersection + 1e-6) / (union + 1e-6)

    def _coarse_binary(self, probs: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(probs, (self.coarse_size, self.coarse_size))
        return (pooled > 0.5).float()

    def forward(self, visual_probs: torch.Tensor, text_prior_probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        entropy = self._binary_entropy(visual_probs).mean(dim=(1, 2, 3))
        confident = entropy < self.entropy_threshold
        agreement = self._dice_score(self._coarse_binary(visual_probs), self._coarse_binary(text_prior_probs))
        agreeing = agreement > self.agreement_threshold
        weight = (confident & agreeing).float().view(-1, 1, 1, 1)
        fused = weight * visual_probs + (1.0 - weight) * text_prior_probs.detach()
        return fused, weight


class BoundaryConsistencyLoss(nn.Module):
    """Encourages boundaries of pseudo-labels and predictions to agree."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.transpose(0, 1)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _edge_map(self, tensor: torch.Tensor) -> torch.Tensor:
        sobel_x = self.sobel_x.to(device=tensor.device, dtype=tensor.dtype)
        sobel_y = self.sobel_y.to(device=tensor.device, dtype=tensor.dtype)
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.float()
        target = target.float()
        pred_edges = self._edge_map(prediction)
        target_edges = self._edge_map(target)
        return F.l1_loss(pred_edges, target_edges)


class PrototypeAlignmentLoss(nn.Module):
    """Aligns visual summaries to the prototype-backed summary used for text-free inference."""

    def forward(self, query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(query, target, dim=-1).mean()
