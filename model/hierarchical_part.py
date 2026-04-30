import sys
sys.path.insert(0, '')

import numpy as np
import torch
import torch.nn as nn

from graph.openpose18_part import AdjMatrixGraph
from model.dynamic_ms_gcn import DynamicMultiScale_GraphConv
from model.ms_gcn import MultiScale_GraphConv
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN


OPENPOSE18_PARTS = [
    [0, 14, 15, 16, 17],
    [1, 2, 5, 8, 11],
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10],
    [11, 12, 13],
]


def build_part_pooling_matrix(num_point=18, parts=OPENPOSE18_PARTS):
    P = np.zeros((len(parts), num_point), dtype=np.float32)
    for part_idx, joint_indices in enumerate(parts):
        weight = 1.0 / float(len(joint_indices))
        for joint_idx in joint_indices:
            P[part_idx, joint_idx] = weight
    return P


def build_part_to_joint_matrix(num_point=18, parts=OPENPOSE18_PARTS):
    Q = np.zeros((num_point, len(parts)), dtype=np.float32)
    for part_idx, joint_indices in enumerate(parts):
        for joint_idx in joint_indices:
            Q[joint_idx, part_idx] = 1.0

    normalizer = Q.sum(axis=1, keepdims=True)
    normalizer[normalizer == 0] = 1.0
    Q = Q / normalizer
    return Q


class PartPooling(nn.Module):
    def __init__(self, num_point=18):
        super().__init__()
        P = build_part_pooling_matrix(num_point=num_point)
        self.register_buffer('P', torch.tensor(P, dtype=torch.float32))

    def forward(self, x):
        # x: (N, C, T, 18)
        # part_x: (N, C, T, 6)
        return torch.einsum('pv,nctv->nctp', self.P.to(dtype=x.dtype), x)


class PartToJointFusion(nn.Module):
    def __init__(self, num_point=18):
        super().__init__()
        Q = build_part_to_joint_matrix(num_point=num_point)
        self.register_buffer('Q', torch.tensor(Q, dtype=torch.float32))

    def forward(self, part_x):
        # part_x: (N, C, T, 6)
        # joint_x: (N, C, T, 18)
        return torch.einsum('vp,nctp->nctv', self.Q.to(dtype=part_x.dtype), part_x)


class HierarchicalPartBranch(nn.Module):
    def __init__(self,
                 channels,
                 use_dynamic_topology=True,
                 num_part_scales=4,
                 dropout=0,
                 activation='relu'):
        super().__init__()

        part_graph = AdjMatrixGraph()
        part_A_binary = part_graph.A_binary

        self.part_pool = PartPooling()
        self.part_to_joint = PartToJointFusion()

        part_gcn_cls = DynamicMultiScale_GraphConv if use_dynamic_topology else MultiScale_GraphConv
        self.part_gcn = part_gcn_cls(
            num_scales=num_part_scales,
            in_channels=channels,
            out_channels=channels,
            A_binary=part_A_binary,
            disentangled_agg=True,
            use_mask=True,
            dropout=dropout,
            activation=activation
        )
        self.part_tcn = MS_TCN(channels, channels)
        self.gamma_part = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (N, C, T, 18)
        part_x = self.part_pool(x)
        part_x = self.part_gcn(part_x)
        part_x = self.part_tcn(part_x)
        joint_part_x = self.part_to_joint(part_x)
        return x + self.gamma_part.to(dtype=x.dtype, device=x.device) * joint_part_x
