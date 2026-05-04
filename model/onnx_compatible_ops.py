import torch


def apply_linear_to_last_dim(x, weight):
    """
    Apply linear projection on the last dim.

    Args:
        x: [..., Vin]
        weight: [Vout, Vin]

    Returns:
        y: [..., Vout]
    """
    if x.dim() < 1:
        raise ValueError(f"x must have at least 1 dim, got shape={tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [Vout, Vin], got shape={tuple(weight.shape)}")

    orig_shape = x.shape
    vin = orig_shape[-1]
    vout = weight.shape[0]

    if weight.shape[1] != vin:
        raise ValueError(f"Input last dim ({vin}) mismatch weight Vin ({weight.shape[1]})")

    weight = weight.to(dtype=x.dtype, device=x.device)
    y = x.reshape(-1, vin)
    y = torch.matmul(y, weight.t())
    y = y.reshape(*orig_shape[:-1], vout)
    return y


def apply_flat_adjacency(x, A):
    """
    Args:
        x: [N, C, T, Vin]
        A: [Vout, Vin]

    Returns:
        y: [N, C, T, Vout]

    Equivalent to torch.einsum('vu,nctu->nctv', A, x)
    """
    return apply_linear_to_last_dim(x, A)


def apply_static_adjacency_multi_scale(x, A_static):
    """
    Args:
        x: [N, C, T, V]
        A_static: [K, V, V]

    Returns:
        y: [N, K, C, T, V]

    Equivalent to torch.einsum('kvu,nctu->nkctv', A_static, x)
    """
    if x.dim() != 4:
        raise ValueError(f"x must be 4D [N,C,T,V], got shape={tuple(x.shape)}")
    if A_static.dim() != 3:
        raise ValueError(f"A_static must be 3D [K,V,V], got shape={tuple(A_static.shape)}")

    N, C, T, V = x.shape
    K = A_static.shape[0]

    if A_static.shape[1] != V or A_static.shape[2] != V:
        raise ValueError(f"A_static shape {tuple(A_static.shape)} incompatible with V={V}")

    A_flat = A_static.reshape(K * V, V)
    y = apply_linear_to_last_dim(x, A_flat)
    y = y.view(N, C, T, K, V)
    y = y.permute(0, 3, 1, 2, 4).contiguous()
    return y


def compute_dynamic_adjacency_matmul(q, k):
    """
    Args:
        q: [N, G, Cg, V]
        k: [N, G, Cg, V]

    Returns:
        A_dyn: [N, G, V, V]

    Equivalent to torch.einsum('ngcv,ngcu->ngvu', q, k)
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k shape mismatch: {tuple(q.shape)} vs {tuple(k.shape)}")
    if q.dim() != 4:
        raise ValueError(f"q/k must be 4D [N,G,Cg,V], got shape={tuple(q.shape)}")

    return torch.matmul(q.transpose(-2, -1), k)


def apply_dynamic_adjacency(x_group, A_dyn):
    """
    Args:
        x_group: [N, G, Cg, T, V]
        A_dyn: [N, G, V, V]

    Returns:
        y: [N, G, Cg, T, V]

    Equivalent to torch.einsum('ngvu,ngctu->ngctv', A_dyn, x_group)
    """
    if x_group.dim() != 5:
        raise ValueError(f"x_group must be 5D [N,G,Cg,T,V], got shape={tuple(x_group.shape)}")
    if A_dyn.dim() != 4:
        raise ValueError(f"A_dyn must be 4D [N,G,V,V], got shape={tuple(A_dyn.shape)}")

    N, G, Cg, T, V = x_group.shape
    if tuple(A_dyn.shape[:2]) != (N, G) or A_dyn.shape[-1] != V or A_dyn.shape[-2] != V:
        raise ValueError(
            f"A_dyn shape {tuple(A_dyn.shape)} incompatible with x_group shape {tuple(x_group.shape)}"
        )

    x_flat = x_group.reshape(N, G, Cg * T, V)
    y = torch.matmul(x_flat, A_dyn.transpose(-1, -2))
    y = y.reshape(N, G, Cg, T, V)
    return y
