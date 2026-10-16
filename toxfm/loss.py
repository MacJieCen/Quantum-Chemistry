"""Losses used by the released SIMG-ParseCaps checkpoints."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import capsule_logits

class CapsuleMarginLoss(nn.Module):
    def __init__(self, m_plus=0.9, m_minus=0.1, negative_weight=0.5):
        super(CapsuleMarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.negative_weight = negative_weight

    def forward(self, norms, targets):
        one_hot = F.one_hot(targets.long(), num_classes=norms.size(1)).float()
        positive = one_hot * F.relu(self.m_plus - norms).square()
        negative = self.negative_weight * (1.0 - one_hot) * F.relu(norms - self.m_minus).square()
        return (positive + negative).sum(dim=1).mean()


class HybridFocalMarginLoss(nn.Module):
    """Capsule margin loss plus class-weighted focal classification loss."""
    def __init__(self, positive_weight, focal_gamma=2.0, margin_weight=0.50, focal_weight=0.50):
        super(HybridFocalMarginLoss, self).__init__()
        self.margin = CapsuleMarginLoss()
        self.positive_weight = float(positive_weight)
        self.focal_gamma = float(focal_gamma)
        self.margin_weight = float(margin_weight)
        self.focal_weight = float(focal_weight)

    def forward(self, norms, targets):
        logits = capsule_logits(norms)
        targets_float = targets.float()
        weights = torch.full_like(targets_float, self.positive_weight)
        per_item = F.binary_cross_entropy_with_logits(
            logits, targets_float, pos_weight=weights, reduction="none"
        )
        probability = torch.sigmoid(logits)
        probability_true = probability * targets_float + (1.0 - probability) * (1.0 - targets_float)
        focal = ((1.0 - probability_true).clamp_min(0.0).pow(self.focal_gamma) * per_item).mean()
        return self.margin_weight * self.margin(norms, targets) + self.focal_weight * focal


class ClassBalancedFocalMarginLoss(nn.Module):
    """Effective-number class balancing with focal and capsule-margin terms."""
    def __init__(self, positive_count, negative_count, beta=0.999, gamma=2.0,
                 margin_weight=0.35, focal_weight=0.65):
        super(ClassBalancedFocalMarginLoss, self).__init__()
        if positive_count < 1 or negative_count < 1:
            raise ValueError("Class-balanced loss requires both classes")
        effective_positive = 1.0 - beta ** float(positive_count)
        effective_negative = 1.0 - beta ** float(negative_count)
        positive_weight = (1.0 - beta) / effective_positive
        negative_weight = (1.0 - beta) / effective_negative
        normalizer = positive_weight + negative_weight
        self.positive_weight = 2.0 * positive_weight / normalizer
        self.negative_weight = 2.0 * negative_weight / normalizer
        self.gamma = float(gamma)
        self.margin_weight = float(margin_weight)
        self.focal_weight = float(focal_weight)
        self.margin = CapsuleMarginLoss()

    def forward(self, norms, targets):
        logits = capsule_logits(norms)
        targets_float = targets.float()
        probability = torch.sigmoid(logits)
        probability_true = probability * targets_float + (1.0 - probability) * (1.0 - targets_float)
        weights = torch.where(
            targets_float > 0.5,
            torch.full_like(targets_float, self.positive_weight),
            torch.full_like(targets_float, self.negative_weight),
        )
        focal = -(weights * (1.0 - probability_true).pow(self.gamma) * torch.log(probability_true.clamp_min(1e-8))).mean()
        return self.margin_weight * self.margin(norms, targets) + self.focal_weight * focal


def make_criterion(loss_name, labels):
    if loss_name == "margin":
        return CapsuleMarginLoss(), {"loss_name": loss_name}
    if loss_name == "hybrid_focal":
        positives = int(np.asarray(labels, dtype=int).sum())
        negatives = int(len(labels) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError("Both classes are required for focal loss weights")
        positive_weight = float(negatives) / float(positives)
        return HybridFocalMarginLoss(positive_weight), {
            "loss_name": loss_name,
            "positive_weight": positive_weight,
            "focal_gamma": 2.0,
            "margin_weight": 0.50,
            "focal_weight": 0.50,
        }
    if loss_name == "class_balanced_focal":
        positives = int(np.asarray(labels, dtype=int).sum())
        negatives = int(len(labels) - positives)
        criterion = ClassBalancedFocalMarginLoss(positives, negatives)
        return criterion, {
            "loss_name": loss_name,
            "positive_count": positives,
            "negative_count": negatives,
            "effective_number_beta": 0.999,
            "focal_gamma": 2.0,
            "margin_weight": 0.35,
            "focal_weight": 0.65,
            "positive_weight": criterion.positive_weight,
            "negative_weight": criterion.negative_weight,
        }
    raise ValueError("Unknown loss {}".format(loss_name))
