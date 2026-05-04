import numpy as np
import torch

from data_gen.convert_openpose18_to_coco17 import convert_openpose18_joint_to_coco17
from graph.coco17 import AdjMatrixGraph as Coco17Graph
from graph.coco17_part import AdjMatrixGraph as Coco17PartGraph
from model.msg3d_plus import Model


def test_coco17_graph_shape():
    graph = Coco17Graph()
    assert graph.A_binary.shape == (17, 17)


def test_coco17_part_graph_shape():
    graph = Coco17PartGraph()
    assert graph.A_binary.shape == (6, 6)


def test_coco17_model_forward_shape():
    model = Model(
        num_class=400,
        num_point=17,
        num_person=2,
        num_gcn_scales=8,
        num_g3d_scales=8,
        graph='graph.coco17.AdjMatrixGraph',
        in_channels=3,
        use_dynamic_topology=True,
        use_part_branch=True,
        num_part_scales=4,
        part_branch_stages=[1, 2, 3],
        skeleton_layout='coco17',
    )
    model.eval()

    x = torch.randn(2, 3, 32, 17, 2)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 400)


def test_openpose18_to_coco17_conversion():
    rng = np.random.RandomState(0)
    openpose_joint = rng.randn(2, 3, 4, 18, 2).astype(np.float32)
    coco_joint = convert_openpose18_joint_to_coco17(openpose_joint)

    assert coco_joint.shape == (2, 3, 4, 17, 2)
    np.testing.assert_allclose(coco_joint[:, :, :, 5, :], openpose_joint[:, :, :, 5, :])
    np.testing.assert_allclose(coco_joint[:, :, :, 6, :], openpose_joint[:, :, :, 2, :])
    np.testing.assert_allclose(coco_joint[:, :, :, 16, :], openpose_joint[:, :, :, 10, :])


if __name__ == '__main__':
    torch.manual_seed(0)
    test_coco17_graph_shape()
    test_coco17_part_graph_shape()
    test_coco17_model_forward_shape()
    test_openpose18_to_coco17_conversion()
    print('All COCO17 layout tests passed.')
