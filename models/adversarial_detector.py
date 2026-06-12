"""
adversarial_detector.py
-----------------------
Defines the AdversarialDetector class: a MobileNetV3-Large backbone
fine-tuned for binary classification of genuine vs. adversarial images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Large_Weights


class AdversarialDetector(nn.Module):
    """
    Binary adversarial sample detector based on MobileNetV3-Large.

    The model classifies an input image as either genuine (class 0) or
    adversarial (class 1). The classification head of the pre-trained
    backbone is replaced with a linear layer producing two logits.

    Freezing strategy (controlled via ``freeze_mode``):
        - ``'none'``         : All parameters are trainable (full fine-tuning).
        - ``'head_only'``    : Only the classification head is trainable;
                            the entire backbone is frozen. Suitable for
                            Phase 1 of a two-phase training schedule.
        - ``'partial'``      : The first layers of the backbone are frozen;
                            the last ``n_unfreeze_blocks`` InvertedResidual
                            blocks and the classification head are trainable.
                            Suitable for Phase 2 of a two-phase schedule.

    Args:
        pretrained (bool): If True, initialises the backbone with
            ImageNet weights. Default: True.
        freeze_mode (str): One of ``'none'``, ``'head_only'``, ``'partial'``.
            Default: ``'none'``.
        n_unfreeze_blocks (int): Number of trailing InvertedResidual blocks
            to unfreeze when ``freeze_mode='partial'``. Ignored otherwise.
            Default: 3.
        device (str or torch.device): Computation device. Default: ``'cpu'``.
    """

    # ImageNet normalisation statistics, consistent with MobileNetV3 pre-training.
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        pretrained: bool = True,
        freeze_mode: str = 'none',
        n_unfreeze_blocks: int = 3,
        device='cpu'
    ):
        super(AdversarialDetector, self).__init__()

        if freeze_mode not in ('none', 'head_only', 'partial'):
            raise ValueError(
                f"Invalid freeze_mode '{freeze_mode}'. "
                "Choose one of: 'none', 'head_only', 'partial'."
            )

        self.device       = torch.device(device)
        self.freeze_mode  = freeze_mode

        # ------------------------------------------------------------------
        # Backbone: MobileNetV3-Large with ImageNet weights
        # ------------------------------------------------------------------
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_large(weights=weights)

        # Replace the final linear layer to produce 2 logits.
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, 2)

        # ------------------------------------------------------------------
        # Apply the requested freezing strategy
        # ------------------------------------------------------------------
        self._apply_freeze(n_unfreeze_blocks)

        # ------------------------------------------------------------------
        # Preprocessing pipeline
        # Input images are expected to be float tensors in [0, 1] at
        # 160x160 resolution. Only ImageNet normalisation is applied;
        # no spatial resizing is needed.
        # ------------------------------------------------------------------
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=self._IMAGENET_MEAN,
                std=self._IMAGENET_STD
            )
        ])

        self.to(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_freeze(self, n_unfreeze_blocks: int) -> None:
        """
        Sets the ``requires_grad`` flag on all parameters according to
        the selected ``freeze_mode``.
        """
        if self.freeze_mode == 'none':
            # All parameters remain trainable.
            for param in self.parameters():
                param.requires_grad = True

        elif self.freeze_mode == 'head_only':
            # Freeze the entire backbone; unfreeze only the classifier head.
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

        elif self.freeze_mode == 'partial':
            # Freeze everything first.
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze the classifier head.
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

            # Unfreeze the last n_unfreeze_blocks InvertedResidual blocks
            # inside backbone.features (a Sequential).
            features = self.backbone.features
            total_blocks = len(features)
            unfreeze_from = max(0, total_blocks - n_unfreeze_blocks)

            for block in features[unfreeze_from:]:
                for param in block.parameters():
                    param.requires_grad = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_freeze_mode(self, freeze_mode: str, n_unfreeze_blocks: int = 3) -> None:
        """
        Changes the freezing strategy at runtime without re-instantiating
        the model. Useful for implementing a two-phase training schedule.

        Args:
            freeze_mode (str): One of ``'none'``, ``'head_only'``, ``'partial'``.
            n_unfreeze_blocks (int): Relevant only when ``freeze_mode='partial'``.
        """
        if freeze_mode not in ('none', 'head_only', 'partial'):
            raise ValueError(
                f"Invalid freeze_mode '{freeze_mode}'. "
                "Choose one of: 'none', 'head_only', 'partial'."
            )
        self.freeze_mode = freeze_mode
        self._apply_freeze(n_unfreeze_blocks)

    def count_trainable_params(self) -> int:
        """
        Returns the number of parameters currently set as trainable.
        Useful for logging the effect of the active freeze strategy.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Expects a batch of images as float tensors in [0, 1].
        ImageNet normalisation is applied internally before passing to the
        backbone.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 160, 160)
                with values in [0, 1].

        Returns:
            torch.Tensor: Raw logits of shape (N, 2).
        """
        x = self.transform(x)
        return self.backbone(x)

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability of the adversarial class (class 1) for
        each image in the batch. No threshold is applied; the caller is
        responsible for any decision logic.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 160, 160)
                with values in [0, 1], already on the correct device.

        Returns:
            torch.Tensor: 1-D tensor of shape (N,) with adversarial
                scores in [0, 1].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
        return probs[:, 1]

    def save_weights(self, path: str) -> None:
        """
        Saves the current model state dictionary to disk.

        Args:
            path (str): Destination file path (e.g. 'models/detector.pth').
        """
        torch.save(self.state_dict(), path)
        print(f"[INFO] Detector weights saved to: {path}")

    def load_weights(self, path: str) -> None:
        """
        Loads a state dictionary from disk into the current model.

        Args:
            path (str): Source file path.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.eval()
        print(f"[INFO] Detector weights loaded from: {path}")