import sys
sys.path.insert(0, '')

import math
import numpy as np
import torch
import torch.nn as nn

from graph.tools import k_adjacency, normalize_adjacency_matrix
from model.mlp import MLP


class DynamicMultiScale_GraphConv(nn.Module):
    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 A_binary,
                 disentangled_agg=True,
                 use_mask=True,
                 dropout=0,
                 activation='relu',
                 dynamic=True,
                 num_groups=4,
                 reduction=4,
                 dynamic_softmax=True):
        super().__init__()
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_mask = use_mask
        self.dynamic = dynamic
        self.dynamic_softmax = dynamic_softmax

        reduction = max(int(reduction), 1)
        self.hidden_channels = max(in_channels // reduction, 8)
        self.num_groups = self._resolve_num_groups(in_channels, num_groups)
        self.group_channels = in_channels // self.num_groups

        A_binary = np.asarray(A_binary)
        num_nodes = A_binary.shape[-1]

        if disentangled_agg:
            A_powers = [k_adjacency(A_binary, k, with_self=True) for k in range(num_scales)]
            A_powers = np.concatenate([normalize_adjacency_matrix(g) for g in A_powers])
        else:
            A_powers = [A_binary + np.eye(len(A_binary), dtype=A_binary.dtype) for k in range(num_scales)]
            A_powers = [normalize_adjacency_matrix(g) for g in A_powers]
            A_powers = [np.linalg.matrix_power(g, k) for k, g in enumerate(A_powers)]
            A_powers = np.concatenate(A_powers)

        A_powers = A_powers.reshape(num_scales, num_nodes, num_nodes)
        self.register_buffer('A_powers', torch.tensor(A_powers, dtype=torch.float32))

        if use_mask:
            self.A_res = nn.Parameter(torch.empty_like(self.A_powers))
            nn.init.uniform_(self.A_res, -1e-6, 1e-6)
        else:
            self.A_res = None

        self.gamma = nn.Parameter(torch.zeros(1))

        if self.dynamic:
            dynamic_channels = self.hidden_channels * self.num_groups
            self.theta = nn.Conv1d(in_channels, dynamic_channels, kernel_size=1)
            self.phi = nn.Conv1d(in_channels, dynamic_channels, kernel_size=1)
        else:
            self.theta = None
            self.phi = None

        self.mlp = MLP(in_channels * num_scales, [out_channels], dropout=dropout, activation=activation)

    @staticmethod
    def _resolve_num_groups(in_channels, requested_groups):
        requested_groups = max(1, min(int(requested_groups), int(in_channels)))
        for groups in range(requested_groups, 0, -1):
            if in_channels % groups == 0:
                return groups
        return 1

    def _compute_dynamic_adjacency(self, x):
        N, _, _, V = x.shape
        feat = x.mean(dim=2)
        q = self.theta(feat).reshape(N, self.num_groups, self.hidden_channels, V)
        k = self.phi(feat).reshape(N, self.num_groups, self.hidden_channels, V)

        A_dyn = torch.einsum('ngcv,ngcu->ngvu', q, k) / math.sqrt(self.hidden_channels)
        A_dyn = torch.where(torch.isfinite(A_dyn), A_dyn, torch.zeros_like(A_dyn))

        if self.dynamic_softmax:
            A_dyn = torch.softmax(A_dyn, dim=-1)
        else:
            A_dyn = torch.tanh(A_dyn)

        return torch.where(torch.isfinite(A_dyn), A_dyn, torch.zeros_like(A_dyn))

    def forward(self, x):
        N, C, T, V = x.shape

        A_static = self.A_powers.to(dtype=x.dtype, device=x.device)
        if self.use_mask:
            A_static = A_static + self.A_res.to(dtype=x.dtype, device=x.device)

        static_support = torch.einsum('kvu,nctu->nkctv', A_static, x)

        if self.dynamic:
            A_dyn = self._compute_dynamic_adjacency(x)
            x_group = x.reshape(N, self.num_groups, self.group_channels, T, V)
            dynamic_support = torch.einsum('ngvu,ngctu->ngctv', A_dyn, x_group)
            dynamic_support = dynamic_support.reshape(N, C, T, V)
            dynamic_support = dynamic_support.unsqueeze(1).expand(-1, self.num_scales, -1, -1, -1)
            gamma = self.gamma.to(dtype=x.dtype, device=x.device)
            support = static_support + gamma * dynamic_support
        else:
            support = static_support

        support = support.contiguous().view(N, self.num_scales * C, T, V)
        out = self.mlp(support)
        return out


if __name__ == "__main__":
    from graph.ntu_rgb_d import AdjMatrixGraph
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    msgcn = DynamicMultiScale_GraphConv(num_scales=15, in_channels=3, out_channels=64, A_binary=A_binary)
    msgcn.forward(torch.randn(16, 3, 30, 25))
