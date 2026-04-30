import torch
import torch.nn as nn


class CrossStreamFusion(nn.Module):
    def __init__(
        self,
        channels,
        reduction=16,
        fusion_type='gated_residual',
        gamma_init=0.0,
    ):
        super().__init__()
        if fusion_type != 'gated_residual':
            raise ValueError("Only fusion_type='gated_residual' is currently supported.")

        hidden_channels = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.gate_joint = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gate_bone = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.gamma_j = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_b = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, joint_feat, bone_feat):
        # joint_feat: (N, C, T, V)
        # bone_feat:  (N, C, T, V)
        if joint_feat.shape != bone_feat.shape:
            raise ValueError('joint_feat and bone_feat must have identical shapes.')

        z = torch.cat([joint_feat, bone_feat], dim=1)
        pooled = self.pool(z)
        hidden = self.shared_mlp(pooled)
        gate_j = torch.sigmoid(self.gate_joint(hidden))
        gate_b = torch.sigmoid(self.gate_bone(hidden))

        gamma_j = self.gamma_j.to(dtype=joint_feat.dtype, device=joint_feat.device)
        gamma_b = self.gamma_b.to(dtype=bone_feat.dtype, device=bone_feat.device)
        joint_out = joint_feat + gamma_j * gate_j * bone_feat
        bone_out = bone_feat + gamma_b * gate_b * joint_feat
        return joint_out, bone_out
