import torch

from model.onnx_compatible_ops import (
    apply_dynamic_adjacency,
    apply_flat_adjacency,
    apply_linear_to_last_dim,
    apply_static_adjacency_multi_scale,
    compute_dynamic_adjacency_matmul,
)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def test_apply_flat_adjacency():
    torch.manual_seed(0)
    N, C, T, V, K = 2, 3, 5, 18, 4
    A = torch.randn(K * V, V, dtype=torch.float64)
    x = torch.randn(N, C, T, V, dtype=torch.float64)
    ref = torch.einsum("vu,nctu->nctv", A, x)
    out = apply_flat_adjacency(x, A)
    assert _max_abs_diff(ref, out) < 1e-6


def test_apply_static_adjacency_multi_scale():
    torch.manual_seed(1)
    N, C, T, V, K = 2, 3, 5, 18, 4
    A_static = torch.randn(K, V, V, dtype=torch.float64)
    x = torch.randn(N, C, T, V, dtype=torch.float64)
    ref = torch.einsum("kvu,nctu->nkctv", A_static, x)
    out = apply_static_adjacency_multi_scale(x, A_static)
    assert _max_abs_diff(ref, out) < 1e-6


def test_part_pooling_equivalence():
    torch.manual_seed(2)
    N, C, T, V, Pnum = 2, 3, 5, 18, 6
    P = torch.randn(Pnum, V, dtype=torch.float64)
    x = torch.randn(N, C, T, V, dtype=torch.float64)
    ref = torch.einsum("pv,nctv->nctp", P, x)
    out = apply_linear_to_last_dim(x, P)
    assert _max_abs_diff(ref, out) < 1e-6


def test_part_to_joint_fusion_equivalence():
    torch.manual_seed(3)
    N, C, T, V, Pnum = 2, 3, 5, 18, 6
    Q = torch.randn(V, Pnum, dtype=torch.float64)
    part_x = torch.randn(N, C, T, Pnum, dtype=torch.float64)
    ref = torch.einsum("vp,nctp->nctv", Q, part_x)
    out = apply_linear_to_last_dim(part_x, Q)
    assert _max_abs_diff(ref, out) < 1e-6


def test_compute_dynamic_adjacency_matmul():
    torch.manual_seed(4)
    N, G, Cg, V = 2, 4, 8, 18
    q = torch.randn(N, G, Cg, V, dtype=torch.float64)
    k = torch.randn(N, G, Cg, V, dtype=torch.float64)
    ref = torch.einsum("ngcv,ngcu->ngvu", q, k)
    out = compute_dynamic_adjacency_matmul(q, k)
    assert _max_abs_diff(ref, out) < 1e-6


def test_apply_dynamic_adjacency():
    torch.manual_seed(5)
    N, G, Cg, T, V = 2, 4, 6, 5, 18
    A_dyn = torch.randn(N, G, V, V, dtype=torch.float64)
    x_group = torch.randn(N, G, Cg, T, V, dtype=torch.float64)
    ref = torch.einsum("ngvu,ngctu->ngctv", A_dyn, x_group)
    out = apply_dynamic_adjacency(x_group, A_dyn)
    assert _max_abs_diff(ref, out) < 1e-6
