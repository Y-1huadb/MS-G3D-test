import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import import_class, count_params
from feeders.bone_utils import make_openpose18_bone
from model.dynamic_ms_gcn import DynamicMultiScale_GraphConv
from model.hierarchical_part import HierarchicalPartBranch
from model.joint_bone_fusion import CrossStreamFusion
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
                 use_joint_bone_mid_fusion=False,
                 bone_input_mode='auto',
                 final_fusion='concat',
                 num_part_scales=4,
                 part_branch_stages=(1, 2, 3),
                 **kwargs):
        super(Model, self).__init__()

        if bone_input_mode not in ('auto', 'external'):
            raise ValueError("bone_input_mode must be 'auto' or 'external'.")
        if final_fusion not in ('concat', 'sum'):
            raise ValueError("final_fusion must be 'concat' or 'sum'.")
        if use_part_branch and num_point != 18:
            raise ValueError('Hierarchical part branch currently only supports OpenPose18 (num_point=18).')
        if use_joint_bone_mid_fusion and num_point != 18:
            raise ValueError('Joint-bone mid-level fusion currently only supports OpenPose18 (num_point=18).')

        Graph = import_class(graph)
        A_binary = Graph().A_binary
        gcn_cls = DynamicMultiScale_GraphConv if use_dynamic_topology else MultiScale_GraphConv

        self.use_dynamic_topology = use_dynamic_topology
        self.use_part_branch = use_part_branch
        self.use_joint_bone_mid_fusion = use_joint_bone_mid_fusion
        self.bone_input_mode = bone_input_mode
        self.final_fusion = final_fusion
        self.part_branch_stages = list(part_branch_stages)
        self.joint_bone_fusion_stages = [1, 2, 3]
        self.num_part_scales = num_part_scales
        self.num_person = num_person

        c1 = 96
        c2 = c1 * 2
        c3 = c2 * 2

        if self.use_joint_bone_mid_fusion:
            self.data_bn_joint = nn.BatchNorm1d(num_person * in_channels * num_point)
            self.data_bn_bone = nn.BatchNorm1d(num_person * in_channels * num_point)
            self.j_gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
            self.j_sgcn1 = nn.Sequential(
                gcn_cls(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
                MS_TCN(c1, c1),
                MS_TCN(c1, c1))
            self.j_sgcn1[-1].act = nn.Identity()
            self.j_tcn1 = MS_TCN(c1, c1)

            self.j_gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
            self.j_sgcn2 = nn.Sequential(
                gcn_cls(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
                MS_TCN(c1, c2, stride=2),
                MS_TCN(c2, c2))
            self.j_sgcn2[-1].act = nn.Identity()
            self.j_tcn2 = MS_TCN(c2, c2)

            self.j_gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
            self.j_sgcn3 = nn.Sequential(
                gcn_cls(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
                MS_TCN(c2, c3, stride=2),
                MS_TCN(c3, c3))
            self.j_sgcn3[-1].act = nn.Identity()
            self.j_tcn3 = MS_TCN(c3, c3)

            self.b_gcn3d1 = MultiWindow_MS_G3D(3, c1, A_binary, num_g3d_scales, window_stride=1)
            self.b_sgcn1 = nn.Sequential(
                gcn_cls(num_gcn_scales, 3, c1, A_binary, disentangled_agg=True),
                MS_TCN(c1, c1),
                MS_TCN(c1, c1))
            self.b_sgcn1[-1].act = nn.Identity()
            self.b_tcn1 = MS_TCN(c1, c1)

            self.b_gcn3d2 = MultiWindow_MS_G3D(c1, c2, A_binary, num_g3d_scales, window_stride=2)
            self.b_sgcn2 = nn.Sequential(
                gcn_cls(num_gcn_scales, c1, c1, A_binary, disentangled_agg=True),
                MS_TCN(c1, c2, stride=2),
                MS_TCN(c2, c2))
            self.b_sgcn2[-1].act = nn.Identity()
            self.b_tcn2 = MS_TCN(c2, c2)

            self.b_gcn3d3 = MultiWindow_MS_G3D(c2, c3, A_binary, num_g3d_scales, window_stride=2)
            self.b_sgcn3 = nn.Sequential(
                gcn_cls(num_gcn_scales, c2, c2, A_binary, disentangled_agg=True),
                MS_TCN(c2, c3, stride=2),
                MS_TCN(c3, c3))
            self.b_sgcn3[-1].act = nn.Identity()
            self.b_tcn3 = MS_TCN(c3, c3)

            if self.use_part_branch:
                self.j_part1 = HierarchicalPartBranch(c1, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
                self.j_part2 = HierarchicalPartBranch(c2, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
                self.j_part3 = HierarchicalPartBranch(c3, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
                self.b_part1 = HierarchicalPartBranch(c1, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
                self.b_part2 = HierarchicalPartBranch(c2, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
                self.b_part3 = HierarchicalPartBranch(c3, use_dynamic_topology=use_dynamic_topology, num_part_scales=num_part_scales)
            else:
                self.j_part1 = None
                self.j_part2 = None
                self.j_part3 = None
                self.b_part1 = None
                self.b_part2 = None
                self.b_part3 = None

            self.fusion1 = CrossStreamFusion(c1, gamma_init=0.0)
            self.fusion2 = CrossStreamFusion(c2, gamma_init=0.0)
            self.fusion3 = CrossStreamFusion(c3, gamma_init=0.0)
            self.fc = nn.Linear(c3 * 2 if self.final_fusion == 'concat' else c3, num_class)
        else:
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
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
        print('[MSG3D-Plus] use_joint_bone_mid_fusion = {}'.format(self.use_joint_bone_mid_fusion))
        print('[MSG3D-Plus] bone_input_mode = {}'.format(self.bone_input_mode))
        print('[MSG3D-Plus] final_fusion = {}'.format(self.final_fusion))
        print('[MSG3D-Plus] part_branch_stages = {}'.format(self.part_branch_stages))
        print('[MSG3D-Plus] num_part_scales = {}'.format(self.num_part_scales))
        print('[MSG3D-Plus] part_gamma_init = 0.0')
        if self.use_joint_bone_mid_fusion:
            print('[MSG3D-Plus] joint-bone fusion stages = {}'.format(self.joint_bone_fusion_stages))
            print('[MSG3D-Plus] fusion_gamma_init = 0.0')
            print('[MSG3D-Plus] using separate data_bn for joint and bone streams')
        else:
            print('[MSG3D-Plus] joint-bone mid-level fusion disabled, using single-stream mode')
        if self.use_dynamic_topology:
            print('[MSG3D-Plus] using DynamicMultiScale_GraphConv')
        else:
            print('[MSG3D-Plus] using official MultiScale_GraphConv')

    def _preprocess_input(self, x, data_bn):
        # x: (N, C, T, V, M)
        n_batch, c_in, t_steps, v_num, n_person = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n_batch, n_person * v_num * c_in, t_steps)
        x = data_bn(x)
        x = x.view(n_batch, n_person, v_num, c_in, t_steps).permute(0, 1, 3, 4, 2).contiguous().view(n_batch * n_person, c_in, t_steps, v_num)
        return x

    def _run_stage(self, x, sgcn, gcn3d, tcn, part_branch=None):
        # x: (N * M, C, T, V)
        x = F.relu(sgcn(x) + gcn3d(x), inplace=True)
        x = tcn(x)
        if part_branch is not None:
            x = part_branch(x)
        return x

    def forward(self, x, x_bone=None):
        n_batch, _, _, _, n_person = x.size()

        if not self.use_joint_bone_mid_fusion:
            x = self._preprocess_input(x, self.data_bn)
            x = self._run_stage(x, self.sgcn1, self.gcn3d1, self.tcn1, self.part1 if self.use_part_branch and 1 in self.part_branch_stages else None)
            x = self._run_stage(x, self.sgcn2, self.gcn3d2, self.tcn2, self.part2 if self.use_part_branch and 2 in self.part_branch_stages else None)
            x = self._run_stage(x, self.sgcn3, self.gcn3d3, self.tcn3, self.part3 if self.use_part_branch and 3 in self.part_branch_stages else None)

            out_channels = x.size(1)
            x = x.view(n_batch, n_person, out_channels, -1)
            x = x.mean(3)
            x = x.mean(1)
            return self.fc(x)

        if x_bone is None:
            if self.bone_input_mode == 'auto':
                x_bone = make_openpose18_bone(x)
            else:
                raise ValueError("x_bone must be provided when bone_input_mode='external'.")

        joint_x = self._preprocess_input(x, self.data_bn_joint)
        bone_x = self._preprocess_input(x_bone, self.data_bn_bone)

        joint_x = self._run_stage(joint_x, self.j_sgcn1, self.j_gcn3d1, self.j_tcn1, self.j_part1 if self.use_part_branch and 1 in self.part_branch_stages else None)
        bone_x = self._run_stage(bone_x, self.b_sgcn1, self.b_gcn3d1, self.b_tcn1, self.b_part1 if self.use_part_branch and 1 in self.part_branch_stages else None)
        joint_x, bone_x = self.fusion1(joint_x, bone_x)

        joint_x = self._run_stage(joint_x, self.j_sgcn2, self.j_gcn3d2, self.j_tcn2, self.j_part2 if self.use_part_branch and 2 in self.part_branch_stages else None)
        bone_x = self._run_stage(bone_x, self.b_sgcn2, self.b_gcn3d2, self.b_tcn2, self.b_part2 if self.use_part_branch and 2 in self.part_branch_stages else None)
        joint_x, bone_x = self.fusion2(joint_x, bone_x)

        joint_x = self._run_stage(joint_x, self.j_sgcn3, self.j_gcn3d3, self.j_tcn3, self.j_part3 if self.use_part_branch and 3 in self.part_branch_stages else None)
        bone_x = self._run_stage(bone_x, self.b_sgcn3, self.b_gcn3d3, self.b_tcn3, self.b_part3 if self.use_part_branch and 3 in self.part_branch_stages else None)
        joint_x, bone_x = self.fusion3(joint_x, bone_x)

        out_channels = joint_x.size(1)
        joint_x = joint_x.view(n_batch, n_person, out_channels, -1).mean(dim=3).mean(dim=1)
        bone_x = bone_x.view(n_batch, n_person, out_channels, -1).mean(dim=3).mean(dim=1)

        if self.final_fusion == 'concat':
            feat = torch.cat([joint_x, bone_x], dim=1)
        else:
            feat = joint_x + bone_x
        return self.fc(feat)


if __name__ == "__main__":
    model = Model(
        num_class=60,
        num_point=18,
        num_person=2,
        num_gcn_scales=8,
        num_g3d_scales=8,
        graph='graph.kinetics.AdjMatrixGraph',
        in_channels=3,
        use_dynamic_topology=True,
        use_part_branch=True,
        use_joint_bone_mid_fusion=True,
        bone_input_mode='auto',
        final_fusion='concat',
        num_part_scales=4,
        part_branch_stages=[1, 2, 3],
    )

    N, C, T, V, M = 2, 3, 64, 18, 2
    x = torch.randn(N, C, T, V, M)
    y = model(x)

    print('Output shape:', y.shape)
    print('Model total # params:', count_params(model))
