import torch

from graph.kinetics import AdjMatrixGraph
from model.dynamic_ms_gcn import DynamicMultiScale_GraphConv
from model.msg3d_dynamic import Model


def test_dynamic_msgcn_input_stage():
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    x = torch.randn(2, 3, 32, 18)
    layer = DynamicMultiScale_GraphConv(
        num_scales=8,
        in_channels=3,
        out_channels=96,
        A_binary=A_binary,
        disentangled_agg=True
    )
    y = layer(x)
    assert y.shape == (2, 96, 32, 18)


def test_dynamic_msgcn_middle_stage():
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    x = torch.randn(2, 96, 32, 18)
    layer = DynamicMultiScale_GraphConv(
        num_scales=8,
        in_channels=96,
        out_channels=96,
        A_binary=A_binary
    )
    y = layer(x)
    assert y.shape == (2, 96, 32, 18)


def test_dynamic_msg3d_model_and_backward():
    model = Model(
        num_class=400,
        num_point=18,
        num_person=2,
        num_gcn_scales=8,
        num_g3d_scales=8,
        graph='graph.kinetics.AdjMatrixGraph',
        in_channels=3
    )
    x = torch.randn(2, 3, 64, 18, 2)
    y = model(x)
    assert y.shape == (2, 400)

    loss = y.mean()
    loss.backward()

    layer = model.sgcn1[0]
    assert isinstance(layer, DynamicMultiScale_GraphConv)
    assert layer.gamma.grad is not None
    assert layer.theta.weight.grad is not None
    assert layer.phi.weight.grad is not None
    assert torch.isfinite(layer.gamma.grad).all()
    assert torch.isfinite(layer.theta.weight.grad).all()
    assert torch.isfinite(layer.phi.weight.grad).all()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_dynamic_msgcn_input_stage()
    test_dynamic_msgcn_middle_stage()
    test_dynamic_msg3d_model_and_backward()
    print('All dynamic MSGCN tests passed.')
