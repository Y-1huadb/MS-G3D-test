import torch

from model.hierarchical_part import HierarchicalPartBranch, PartPooling, PartToJointFusion
from model.msg3d_plus import Model


def test_part_pooling_shape():
    x = torch.randn(2, 96, 32, 18)
    pool = PartPooling()
    y = pool(x)
    assert y.shape == (2, 96, 32, 6)


def test_part_to_joint_fusion_shape():
    x = torch.randn(2, 96, 32, 6)
    up = PartToJointFusion()
    y = up(x)
    assert y.shape == (2, 96, 32, 18)


def test_hierarchical_part_branch_shape():
    x = torch.randn(2, 96, 32, 18)
    branch = HierarchicalPartBranch(
        channels=96,
        use_dynamic_topology=True,
        num_part_scales=4
    )
    y = branch(x)
    assert y.shape == (2, 96, 32, 18)


def test_gamma_part_zero_equivalence():
    x = torch.randn(2, 96, 32, 18)
    branch = HierarchicalPartBranch(
        channels=96,
        use_dynamic_topology=True,
        num_part_scales=4
    )
    branch.eval()
    with torch.no_grad():
        y = branch(x)
    diff = (y - x).abs().max().item()
    assert diff < 1e-5


def test_msg3d_plus_model_and_backward():
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
        num_part_scales=4,
        part_branch_stages=[1, 2, 3]
    )
    x = torch.randn(2, 3, 64, 18, 2)
    y = model(x)
    assert y.shape == (2, 400)

    loss = y.mean()
    loss.backward()

    assert model.part1.gamma_part.grad is not None
    assert model.part2.gamma_part.grad is not None
    assert model.part3.gamma_part.grad is not None
    assert torch.isfinite(model.part1.gamma_part.grad).all()
    assert torch.isfinite(model.part2.gamma_part.grad).all()
    assert torch.isfinite(model.part3.gamma_part.grad).all()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_part_pooling_shape()
    test_part_to_joint_fusion_shape()
    test_hierarchical_part_branch_shape()
    test_gamma_part_zero_equivalence()
    test_msg3d_plus_model_and_backward()
    print('All MSG3D-Plus part branch tests passed.')
