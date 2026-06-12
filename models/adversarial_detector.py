"""
adversarial_detector.py
-----------------------
Defines two adversarial sample detector classes for binary classification
of genuine vs. adversarial images:

    - AdversarialDetector  : MobileNetV3-Large backbone.
    - EfficientNetDetector : EfficientNet-B0 backbone.

Both classes share an identical public interface and apply ImageNet
normalisation internally during the forward pass. All parameters are
trainable (full fine-tuning); no freezing logic is included.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights
)


class AdversarialDetector(nn.Module):
    """
    Binary adversarial sample detector based on MobileNetV3-Large.

    Classifies an input image as genuine (class 0) or adversarial (class 1).
    The classification head is replaced with a two-output linear layer.
    ImageNet normalisation is applied internally; the caller must supply
    float tensors in [0, 1] at 160x160 resolution.

    Args:
        pretrained (bool): If True, initialises the backbone with
            ImageNet weights. Default: True.
        device (str or torch.device): Computation device. Default: 'cpu'.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, pretrained: bool = True, device='cpu'):
        super(AdversarialDetector, self).__init__()

        self.device = torch.device(device)

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 2)

        self.transform = transforms.Normalize(
            mean=self._IMAGENET_MEAN,
            std=self._IMAGENET_STD
        )

        self.to(self.device)

    def count_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Applies ImageNet normalisation internally.

        Args:
            x (torch.Tensor): Float tensor of shape (N, 3, 160, 160)
                with values in [0, 1].

        Returns:
            torch.Tensor: Raw logits of shape (N, 2).
        """
        return self.backbone(self.transform(x))

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the adversarial class probability for each image.
        No threshold is applied.

        Args:
            x (torch.Tensor): Float tensor of shape (N, 3, 160, 160)
                with values in [0, 1], already on the correct device.

        Returns:
            torch.Tensor: 1-D tensor of shape (N,) with scores in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)
        return probs[:, 1]

    def save_weights(self, path: str) -> None:
        """Saves the model state dictionary to disk."""
        torch.save(self.state_dict(), path)
        print(f"[INFO] Weights saved to: {path}")

    def load_weights(self, path: str) -> None:
        """Loads a state dictionary from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"[INFO] Weights loaded from: {path}")


class EfficientNetDetector(nn.Module):
    """
    Binary adversarial sample detector based on EfficientNet-B0.

    Identical public interface to AdversarialDetector. The classifier
    head is replaced with a two-output linear layer. ImageNet
    normalisation is applied internally; the caller must supply float
    tensors in [0, 1] at 160x160 resolution.

    Args:
        pretrained (bool): If True, initialises the backbone with
            ImageNet weights. Default: True.
        device (str or torch.device): Computation device. Default: 'cpu'.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, pretrained: bool = True, device='cpu'):
        super(EfficientNetDetector, self).__init__()

        self.device = torch.device(device)

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, 2)

        self.transform = transforms.Normalize(
            mean=self._IMAGENET_MEAN,
            std=self._IMAGENET_STD
        )

        self.to(self.device)

    def count_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Applies ImageNet normalisation internally.

        Args:
            x (torch.Tensor): Float tensor of shape (N, 3, 160, 160)
                with values in [0, 1].

        Returns:
            torch.Tensor: Raw logits of shape (N, 2).
        """
        return self.backbone(self.transform(x))

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the adversarial class probability for each image.
        No threshold is applied.

        Args:
            x (torch.Tensor): Float tensor of shape (N, 3, 160, 160)
                with values in [0, 1], already on the correct device.

        Returns:
            torch.Tensor: 1-D tensor of shape (N,) with scores in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            probs = F.softmax(self.forward(x), dim=1)
        return probs[:, 1]

    def save_weights(self, path: str) -> None:
        """Saves the model state dictionary to disk."""
        torch.save(self.state_dict(), path)
        print(f"[INFO] Weights saved to: {path}")

    def load_weights(self, path: str) -> None:
        """Loads a state dictionary from disk."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"[INFO] Weights loaded from: {path}")