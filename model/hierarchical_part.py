import sys
sys.path.insert(0, '')

import numpy as np
import torch
import torch.nn as nn

from model.dynamic_ms_gcn import DynamicMultiScale_GraphConv
from model.ms_gcn import MultiScale_GraphConv
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.onnx_compatible_ops import apply_linear_to_last_dim
from utils import import_class


OPENPOSE18_PARTS = [
    [0, 14, 15, 16, 17],
    [1, 2, 5, 8, 11],
    [2, 3, 4],
    [5, 6, 7],
    [8, 9, 10],
    [11, 12, 13],
]


COCO17_PARTS = [
    [0, 1, 2, 3, 4],
    [5, 6, 11, 12],
    [6, 8, 10],
    [5, 7, 9],
    [12, 14, 16],
    [11, 13, 15],
]


PARTS_BY_LAYOUT = {
    'openpose18': OPENPOSE18_PARTS,
    'coco17': COCO17_PARTS,
}


PART_GRAPH_BY_LAYOUT = {
    'openpose18': 'graph.openpose18_part.AdjMatrixGraph',
    'coco17': 'graph.coco17_part.AdjMatrixGraph',
}


NUM_POINT_BY_LAYOUT = {
    'openpose18': 18,
    'coco17': 17,
}


def _validate_parts(num_point, parts):
    if not parts:
        raise ValueError('parts must be a non-empty sequence of joint index groups.')

    for part_idx, joint_indices in enumerate(parts):
        if not joint_indices:
            raise ValueError('parts[{}] must not be empty.'.format(part_idx))
        for joint_idx in joint_indices:
            if joint_idx < 0 or joint_idx >= num_point:
                raise ValueError(
                    'Joint index {} in parts[{}] is out of range for num_point={}.'.format(
                        joint_idx, part_idx, num_point
                    )
                )


def _resolve_layout_parts(skeleton_layout, num_point):
    if skeleton_layout not in PARTS_BY_LAYOUT:
        raise ValueError(
            "Unsupported skeleton_layout={!r}. Expected one of: {}.".format(
                skeleton_layout, sorted(PARTS_BY_LAYOUT.keys())
            )
        )

    expected_num_point = NUM_POINT_BY_LAYOUT[skeleton_layout]
    if num_point != expected_num_point:
        raise ValueError(
            'skeleton_layout={} expects num_point={}, but got {}.'.format(
                skeleton_layout, expected_num_point, num_point
            )
        )

    return PARTS_BY_LAYOUT[skeleton_layout]


def _resolve_part_graph(part_graph, skeleton_layout):
    part_graph_ref = part_graph or PART_GRAPH_BY_LAYOUT[skeleton_layout]
    if isinstance(part_graph_ref, str):
        part_graph_ref = import_class(part_graph_ref)

    part_graph_obj = part_graph_ref() if callable(part_graph_ref) else part_graph_ref
    if not hasattr(part_graph_obj, 'A_binary'):
        raise TypeError('part_graph must resolve to an object with an A_binary attribute.')

    return part_graph_obj


def build_part_pooling_matrix(num_point=18, parts=None):
    parts = OPENPOSE18_PARTS if parts is None else parts
    _validate_parts(num_point, parts)

    P = np.zeros((len(parts), num_point), dtype=np.float32)
    for part_idx, joint_indices in enumerate(parts):
        weight = 1.0 / float(len(joint_indices))
        for joint_idx in joint_indices:
            P[part_idx, joint_idx] = weight
    return P


def build_part_to_joint_matrix(num_point=18, parts=None):
    parts = OPENPOSE18_PARTS if parts is None else parts
    _validate_parts(num_point, parts)

    Q = np.zeros((num_point, len(parts)), dtype=np.float32)
    for part_idx, joint_indices in enumerate(parts):
        for joint_idx in joint_indices:
            Q[joint_idx, part_idx] = 1.0

    normalizer = Q.sum(axis=1, keepdims=True)
    normalizer[normalizer == 0] = 1.0
    Q = Q / normalizer
    return Q


class PartPooling(nn.Module):
    def __init__(self, num_point=18, parts=None):
        super().__init__()
        P = build_part_pooling_matrix(num_point=num_point, parts=parts)
        self.register_buffer('P', torch.tensor(P, dtype=torch.float32))

    def forward(self, x):
        # x: (N, C, T, V)
        # part_x: (N, C, T, 6)
        P = self.P.to(dtype=x.dtype, device=x.device)
        return apply_linear_to_last_dim(x, P)


class PartToJointFusion(nn.Module):
    def __init__(self, num_point=18, parts=None):
        super().__init__()
        Q = build_part_to_joint_matrix(num_point=num_point, parts=parts)
        self.register_buffer('Q', torch.tensor(Q, dtype=torch.float32))

    def forward(self, part_x):
        # part_x: (N, C, T, 6)
        # joint_x: (N, C, T, V)
        Q = self.Q.to(dtype=part_x.dtype, device=part_x.device)
        return apply_linear_to_last_dim(part_x, Q)


class HierarchicalPartBranch(nn.Module):
    def __init__(self,
                 channels,
                 num_point=18,
                 skeleton_layout='openpose18',
                 part_graph=None,
                 use_dynamic_topology=True,
                 num_part_scales=4,
                 dropout=0,
                 activation='relu'):
        super().__init__()

        parts = _resolve_layout_parts(skeleton_layout=skeleton_layout, num_point=num_point)
        part_graph = _resolve_part_graph(part_graph=part_graph, skeleton_layout=skeleton_layout)
        part_A_binary = part_graph.A_binary

        self.part_pool = PartPooling(num_point=num_point, parts=parts)
        self.part_to_joint = PartToJointFusion(num_point=num_point, parts=parts)

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
        # x: (N, C, T, V)
        part_x = self.part_pool(x)
        part_x = self.part_gcn(part_x)
        part_x = self.part_tcn(part_x)
        joint_part_x = self.part_to_joint(part_x)
        return x + self.gamma_part.to(dtype=x.dtype, device=x.device) * joint_part_x


