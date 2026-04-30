import torch


OPENPOSE18_INWARD = [
    (4, 3),
    (3, 2),
    (7, 6),
    (6, 5),
    (13, 12),
    (12, 11),
    (10, 9),
    (9, 8),
    (11, 5),
    (8, 2),
    (5, 1),
    (2, 1),
    (0, 1),
    (15, 0),
    (14, 0),
    (17, 15),
    (16, 14),
]


def make_openpose18_bone(x_joint: torch.Tensor) -> torch.Tensor:
    """
    x_joint: (N, 3, T, 18, M)
    return: x_bone: (N, 3, T, 18, M)
    """
    if x_joint.dim() != 5:
        raise ValueError('x_joint must have shape (N, C, T, V, M).')
    if x_joint.size(3) != 18:
        raise ValueError('make_openpose18_bone only supports V=18.')
    if x_joint.size(1) < 3:
        raise ValueError('make_openpose18_bone expects at least 3 channels (x, y, confidence).')

    x_bone = torch.zeros_like(x_joint)
    for child, parent in OPENPOSE18_INWARD:
        x_bone[:, 0:2, :, child, :] = x_joint[:, 0:2, :, child, :] - x_joint[:, 0:2, :, parent, :]
        x_bone[:, 2, :, child, :] = x_joint[:, 2, :, child, :]
    return x_bone
