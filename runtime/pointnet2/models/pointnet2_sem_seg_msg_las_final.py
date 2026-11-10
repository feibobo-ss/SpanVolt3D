import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstractionMsg
from pointcept.models.losses.lovasz import LovaszLoss


class get_model(nn.Module):
    """Frozen LAS PointNet++ MSG; opt-in XYZ-only input preserves the RGB default."""

    def __init__(self, num_classes, in_channels=6):
        super().__init__()
        if in_channels not in (3, 6):
            raise ValueError(f"Expected XYZ (3) or XYZ+RGB (6), got {in_channels}")
        self.in_channels = in_channels
        scales = 2
        self.sa1 = PointNetSetAbstractionMsg(
            1024,
            [0.2 * scales, 0.4 * scales],
            [32, 64],
            in_channels,
            [[16, 16, 32], [32, 32, 64]],
        )
        self.sa2 = PointNetSetAbstractionMsg(
            256,
            [0.4 * scales, 0.8 * scales],
            [32, 64],
            32 + 64,
            [[64, 64, 128], [64, 96, 128]],
        )
        self.sa3 = PointNetSetAbstractionMsg(
            64,
            [0.8 * scales, 1.6 * scales],
            [32, 64],
            128 + 128,
            [[128, 196, 256], [128, 196, 256]],
        )
        self.sa4 = PointNetSetAbstractionMsg(
            16,
            [1.6 * scales, 3.2 * scales],
            [32, 64],
            256 + 256,
            [[256, 256, 512], [256, 384, 512]],
        )
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.6)
        self.conv1b = nn.Conv1d(128, 64, 1)
        self.bn1b = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, num_classes, 1)

    def forward(self, inputs):
        if inputs.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {inputs.shape[1]}")
        points0 = inputs
        xyz0 = inputs[:, :3, :]
        xyz1, points1 = self.sa1(xyz0, points0)
        xyz2, points2 = self.sa2(xyz1, points1)
        xyz3, points3 = self.sa3(xyz2, points2)
        xyz4, points4 = self.sa4(xyz3, points3)
        points3 = self.fp4(xyz3, xyz4, points3, points4)
        points2 = self.fp3(xyz2, xyz3, points2, points3)
        points1 = self.fp2(xyz1, xyz2, points1, points2)
        points0 = self.fp1(xyz0, xyz1, None, points1)
        logits = self.drop1(F.relu(self.bn1(self.conv1(points0))))
        logits = self.drop2(F.relu(self.bn1b(self.conv1b(logits))))
        logits = self.conv2(logits)
        return logits.transpose(1, 2).contiguous()


class get_loss(nn.Module):
    """The benchmark-wide unweighted CE + Lovasz objective."""

    def __init__(self):
        super().__init__()
        self.lovasz = LovaszLoss(mode="multiclass", ignore_index=-1)

    def forward(self, logits, target):
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = target.reshape(-1)
        ce = F.cross_entropy(flat_logits, flat_target, ignore_index=-1)
        return ce + self.lovasz(flat_logits, flat_target)
