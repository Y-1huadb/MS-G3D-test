import torch

from feeders.bone_utils import make_openpose18_bone
from model.joint_bone_fusion import CrossStreamFusion
from model.msg3d_plus import Model


def test_make_openpose18_bone_shape_device_dtype():
    x = torch.randn(2, 3, 32, 18, 2)
    bone = make_openpose18_bone(x)
    assert bone.shape == x.shape
    assert bone.device == x.device
    assert bone.dtype == x.dtype


def test_cross_stream_fusion_shape():
    fusion = CrossStreamFusion(channels=96, gamma_init=0.0)
    j = torch.randn(2, 96, 32, 18)
    b = torch.randn(2, 96, 32, 18)
    j2, b2 = fusion(j, b)
    assert j2.shape == j.shape
    assert b2.shape == b.shape


def test_cross_stream_fusion_identity_at_init():
    fusion = CrossStreamFusion(channels=96, gamma_init=0.0)
    j = torch.randn(2, 96, 32, 18)
    b = torch.randn(2, 96, 32, 18)
    fusion.eval()
    with torch.no_grad():
        j2, b2 = fusion(j, b)
    assert (j2 - j).abs().max().item() < 1e-5
    assert (b2 - b).abs().max().item() < 1e-5


def test_msg3d_plus_joint_bone_model_forward():
    model = Model(
        num_class=400,
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
    x = torch.randn(2, 3, 64, 18, 2)
    y = model(x)
    assert y.shape == (2, 400)


def test_msg3d_plus_forward_with_external_bone():
    model = Model(
        num_class=400,
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
    x = torch.randn(2, 3, 64, 18, 2)
    bone = make_openpose18_bone(x)
    y = model(x, x_bone=bone)
    assert y.shape == (2, 400)


def test_msg3d_plus_joint_bone_backward():
    model = Model(
        num_class=400,
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
    x = torch.randn(2, 3, 64, 18, 2)
    y = model(x)
    loss = y.mean()
    loss.backward()

    assert model.fusion1.gamma_j.grad is not None
    assert model.fusion1.gamma_b.grad is not None
    assert torch.isfinite(model.fusion1.gamma_j.grad).all()
    assert torch.isfinite(model.fusion1.gamma_b.grad).all()

    assert model.j_part1.gamma_part.grad is not None
    assert torch.isfinite(model.j_part1.gamma_part.grad).all()

    assert model.j_sgcn1[0].gamma.grad is not None
    assert model.b_sgcn1[0].gamma.grad is not None
    assert torch.isfinite(model.j_sgcn1[0].gamma.grad).all()
    assert torch.isfinite(model.b_sgcn1[0].gamma.grad).all()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_make_openpose18_bone_shape_device_dtype()
    test_cross_stream_fusion_shape()
    test_cross_stream_fusion_identity_at_init()
    test_msg3d_plus_joint_bone_model_forward()
    test_msg3d_plus_forward_with_external_bone()
    test_msg3d_plus_joint_bone_backward()
    print('All joint-bone mid-fusion tests passed.')
