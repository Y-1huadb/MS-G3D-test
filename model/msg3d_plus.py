import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from model.dynamic_ms_gcn import DynamicMultiScale_GraphConv
from model.hierarchical_part import HierarchicalPartBranch
from model.ms_gcn import MultiScale_GraphConv
from model.ms_tcn import MultiScale_TemporalConv as MS_TCN
from model.msg3d import MultiWindow_MS_G3D


class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_point,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 graph,
                 in_channels=3,
                 use_dynamic_topology=True,
                 use_part_branch=True,
                 num_part_scales=4,
                 part_branch_stages=(1, 2, 3),
                 **kwargs):
        super(Model, self).__init__()

        if use_part_branch and num_point != 18:
            raise ValueError('Hierarchical part branch currently only supports OpenPose18 (num_point=18).')

        Graph = import_class(graph)
        A_binary = Graph().A_binary
        gcn_cls = DynamicMultiScale_GraphConv if use_dynamic_topology else MultiScale_GraphConv

        self.use_dynamic_topology = use_dynamic_topology
        self.use_part_branch = use_part_branch
        self.part_branch_stages = list(part_branch_stages)
        self.num_part_scales = num_part_scales

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1 = 96
        c2 = c1 * 2
        c3 = c2 * 2

        self.gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
        self.sgcn1 = nn.Sequential(
            gcn_cls(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn2 = nn.Sequential(
            gcn_cls(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
        self.sgcn3 = nn.Sequential(
            gcn_cls(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        if self.use_part_branch:
            self.part1 = HierarchicalPartBranch(c1, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
            self.part2 = HierarchicalPartBranch(c2, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
            self.part3 = HierarchicalPartBranch(c3, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
        else:
            self.part1 = None
            self.part2 = None
            self.part3 = None

        self.fc = nn.Linear(c3, num_class)

        print('[MSG3D-Plus] use_dynamic_topology = {}'.format(self.use_dynamic_topology))
        print('[MSG3D-Plus] use_part_branch = {}'.format(self.use_part_branch))
        print('[MSG3D-Plus] part_branch_stages = {}'.format(self.part_branch_stages))
        print('[MSG3D-Plus] num_part_scales = {}'.format(self.num_part_scales))
        print('[MSG3D-Plus] part_gamma_init = 0.0')
        if self.use_dynamic_topology:
            print('[MSG3D-Plus] using DynamicMultiScale_GraphConv')
        else:
            print('[MSG3D-Plus] using official MultiScale_GraphConv')

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()

        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)
        if self.use_part_branch and 1 in self.part_branch_stages:
            x = self.part1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
        if self.use_part_branch and 2 in self.part_branch_stages:
            x = self.part2(x)

        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)
        if self.use_part_branch and 3 in self.part_branch_stages:
            x = self.part3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)
        out = out.mean(1)

        out = self.fc(out)
        return out


if __name__ == "__main__":
    model = Model(
        num_class=60,
        num_point=18,
        num_person=2,
        num_gcn_scales=13,
        num_g3d_scales=6,
        graph='graph.kinetics.AdjMatrixGraph'
    )

    N, C, T, V, M = 6, 3, 50, 18, 2
    x = torch.randn(N, C, T, V, M)
    model.forward(x)

    print('Model total # params:', count_params(model))
