"""
models/encoder.py -- Frozen ResNet18 feature extractor.

Only the stem + layer1 + layer2 are used, giving a 128-channel feature map at
1/8 resolution. All parameters are frozen: GATv2 consumes these features but
never trains them, so the encoder contributes 0 trainable parameters to the
hybrid model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cxr_gnn.utils import get_logger

logger = get_logger(__name__)


def build_encoder(device: torch.device, pretrained: bool = True) -> nn.Module:
    """Build the frozen ResNet18 encoder.

    Args:
        pretrained: use ImageNet weights. Set False for the
                    "no ImageNet pretraining" ablation arm.
    """
    from torchvision.models import resnet18

    backbone = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            logger.info("Loaded ImageNet-pretrained ResNet18.")
        except Exception as ex:
            logger.warning("Pretrained weights unavailable (%s). Using random init.", ex)
    if backbone is None:
        backbone = resnet18(weights=None)
        if pretrained:
            logger.warning("Falling back to randomly initialised ResNet18.")
        else:
            logger.info("Randomly initialised ResNet18 (no-pretraining ablation).")

    encoder = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,      # -> (B, 128, H/8, W/8)
    ).to(device).eval()

    for p in encoder.parameters():
        p.requires_grad = False

    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info("Encoder frozen: %d params (not trained)", n_params)

    return encoder
