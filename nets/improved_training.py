from contextlib import nullcontext
from typing import Dict, Optional

import torch
import torch.nn as nn

from .improved_ssl import BoundaryConsistencyLoss, PrototypeAlignmentLoss, UncertaintyAwarePseudoLabelFusion


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim - 1:
            target = target.unsqueeze(1)
        prediction = torch.sigmoid(logits).float().reshape(logits.shape[0], -1)
        target = target.float().reshape(target.shape[0], -1)
        intersection = (prediction * target).sum(dim=1)
        union = prediction.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class ImprovedSemiSupervisedLoss(nn.Module):
    """Composite objective matching the improved report-driven design."""

    def __init__(
        self,
        lambda_consistency: float = 1.0,
        lambda_text: float = 0.5,
        lambda_boundary: float = 0.25,
        lambda_prototype: float = 0.1,
        entropy_threshold: float = 0.35,
        agreement_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.supervised_bce = nn.BCEWithLogitsLoss()
        self.supervised_dice = SoftDiceLoss()
        self.consistency_bce = nn.BCEWithLogitsLoss()
        self.fusion = UncertaintyAwarePseudoLabelFusion(
            entropy_threshold=entropy_threshold,
            agreement_threshold=agreement_threshold,
        )
        self.boundary = BoundaryConsistencyLoss()
        self.prototype_alignment = PrototypeAlignmentLoss()
        self.lambda_consistency = lambda_consistency
        self.lambda_text = lambda_text
        self.lambda_boundary = lambda_boundary
        self.lambda_prototype = lambda_prototype

    def forward(
        self,
        labeled_logits: torch.Tensor,
        labeled_target: torch.Tensor,
        unlabeled_logits: Optional[torch.Tensor] = None,
        visual_teacher_prediction: Optional[torch.Tensor] = None,
        text_prior_prediction: Optional[torch.Tensor] = None,
        visual_summary: Optional[torch.Tensor] = None,
        prototype_summary: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if labeled_target.ndim == labeled_logits.ndim - 1:
            labeled_target = labeled_target.unsqueeze(1)

        if unlabeled_logits is not None and visual_teacher_prediction is not None:
            if visual_teacher_prediction.ndim == unlabeled_logits.ndim - 1:
                visual_teacher_prediction = visual_teacher_prediction.unsqueeze(1)

        if unlabeled_logits is not None and text_prior_prediction is not None:
            if text_prior_prediction.ndim == unlabeled_logits.ndim - 1:
                text_prior_prediction = text_prior_prediction.unsqueeze(1)

        supervised = self.supervised_bce(labeled_logits, labeled_target.float()) + self.supervised_dice(
            labeled_logits,
            labeled_target.float(),
        )

        consistency = supervised.new_tensor(0.0)
        text_guidance = supervised.new_tensor(0.0)
        boundary = supervised.new_tensor(0.0)
        prototype = supervised.new_tensor(0.0)
        fused_pseudo = None
        fusion_weight = supervised.new_tensor(0.0)

        if (
            unlabeled_logits is not None
            and visual_teacher_prediction is not None
            and text_prior_prediction is not None
        ):
            fused_pseudo, weight = self.fusion(visual_teacher_prediction, text_prior_prediction)
            consistency = self.consistency_bce(unlabeled_logits, fused_pseudo.detach())
            text_guidance = self.consistency_bce(unlabeled_logits, text_prior_prediction.detach())
            boundary = self.boundary(torch.sigmoid(unlabeled_logits), fused_pseudo.detach())
            fusion_weight = weight.mean()

        if visual_summary is not None and prototype_summary is not None:
            prototype = self.prototype_alignment(visual_summary, prototype_summary.detach())

        total = (
            supervised
            + self.lambda_consistency * consistency
            + self.lambda_text * text_guidance
            + self.lambda_boundary * boundary
            + self.lambda_prototype * prototype
        )

        return {
            "total": total,
            "supervised": supervised,
            "consistency": consistency,
            "text_guidance": text_guidance,
            "boundary": boundary,
            "prototype": prototype,
            "fusion_weight": fusion_weight,
            "fused_pseudo": fused_pseudo if fused_pseudo is not None else torch.sigmoid(labeled_logits.detach()),
        }


@torch.no_grad()
def update_ema_model(teacher: nn.Module, student: nn.Module, decay: float = 0.99) -> None:
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


class ImprovedSSLTrainer:
    """Minimal SSL-native trainer that wires the improved model into a dual-teacher loop."""

    def __init__(
        self,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        teacher: Optional[nn.Module] = None,
        loss_fn: Optional[ImprovedSemiSupervisedLoss] = None,
        ema_decay: float = 0.99,
        student_view_noise_std: float = 0.05,
        student_view_dropout_prob: float = 0.0,
    ) -> None:
        self.student = student
        self.teacher = teacher if teacher is not None else self._clone_teacher(student)
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else ImprovedSemiSupervisedLoss()
        self.ema_decay = ema_decay
        self.student_view_noise_std = student_view_noise_std
        self.student_view_dropout_prob = student_view_dropout_prob

    @staticmethod
    def _clone_teacher(student: nn.Module) -> nn.Module:
        import copy

        teacher = copy.deepcopy(student)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        return teacher

    def _make_student_view(self, images: torch.Tensor) -> torch.Tensor:
        augmented = images.clone()
        if self.student_view_noise_std > 0:
            augmented = augmented + torch.randn_like(augmented) * self.student_view_noise_std
        if self.student_view_dropout_prob > 0:
            dropout_mask = torch.rand_like(augmented[:, :1]) > self.student_view_dropout_prob
            augmented = augmented * dropout_mask
        augmented = augmented.clamp(0.0, 1.0)
        return augmented

    def training_step(
        self,
        labeled_images: torch.Tensor,
        labeled_masks: torch.Tensor,
        labeled_text_tokens: Optional[torch.Tensor] = None,
        labeled_structured_attributes: Optional[torch.Tensor] = None,
        unlabeled_images: Optional[torch.Tensor] = None,
        unlabeled_teacher_images: Optional[torch.Tensor] = None,
        unlabeled_student_images: Optional[torch.Tensor] = None,
        unlabeled_text_tokens: Optional[torch.Tensor] = None,
        unlabeled_structured_attributes: Optional[torch.Tensor] = None,
        unlabeled_force_text_dropout_mask: Optional[torch.Tensor] = None,
        scaler: Optional[torch.amp.GradScaler] = None,
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> Dict[str, torch.Tensor]:
        self.student.train()
        device_type = labeled_images.device.type
        amp_enabled = use_amp and device_type == "cuda"
        autocast_context = torch.amp.autocast(device_type="cuda", enabled=True) if amp_enabled else nullcontext()

        with autocast_context:
            labeled_logits, labeled_aux = self.student(
                labeled_images,
                text_tokens=labeled_text_tokens,
                structured_attributes=labeled_structured_attributes,
                return_aux=True,
                update_prototypes=True,
            )

            unlabeled_logits = None
            teacher_prediction = None
            teacher_text_prior = None
            unlabeled_aux = None
            if unlabeled_images is not None:
                teacher_images = unlabeled_teacher_images if unlabeled_teacher_images is not None else unlabeled_images
                student_images = (
                    unlabeled_student_images if unlabeled_student_images is not None else self._make_student_view(unlabeled_images)
                )
                unlabeled_logits, unlabeled_aux = self.student(
                    student_images,
                    text_tokens=unlabeled_text_tokens,
                    structured_attributes=unlabeled_structured_attributes,
                    return_aux=True,
                    update_prototypes=False,
                    force_text_dropout_mask=unlabeled_force_text_dropout_mask,
                )
                with torch.no_grad():
                    self.teacher.eval()
                    teacher_logits, teacher_aux = self.teacher(
                        teacher_images,
                        text_tokens=unlabeled_text_tokens,
                        structured_attributes=unlabeled_structured_attributes,
                        return_aux=True,
                        update_prototypes=False,
                    )
                    teacher_prediction = torch.sigmoid(teacher_logits)
                    teacher_text_prior = teacher_aux["prior_prediction"]

            losses = self.loss_fn(
                labeled_logits=labeled_logits,
                labeled_target=labeled_masks,
                unlabeled_logits=unlabeled_logits,
                visual_teacher_prediction=teacher_prediction,
                text_prior_prediction=teacher_text_prior,
                visual_summary=unlabeled_aux["fused_summary"] if unlabeled_aux is not None else labeled_aux["fused_summary"],
                prototype_summary=(
                    unlabeled_aux["prototype_summary"] if unlabeled_aux is not None else labeled_aux["prototype_summary"]
                ),
            )

        if zero_grad:
            self.optimizer.zero_grad(set_to_none=True)

        backward_loss = losses["total"] / max(int(grad_accum_steps), 1)
        if scaler is not None and amp_enabled:
            scaler.scale(backward_loss).backward()
        else:
            backward_loss.backward()

        if step_optimizer:
            if scaler is not None and amp_enabled:
                scaler.step(self.optimizer)
                scaler.update()
            else:
                self.optimizer.step()
            update_ema_model(self.teacher, self.student, decay=self.ema_decay)

        losses["backward_loss"] = backward_loss.detach()
        return losses
