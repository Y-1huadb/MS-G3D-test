import argparse
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


# OpenPose18 neck (index 1) is intentionally dropped because COCO17 has no neck joint.
OPENPOSE18_TO_COCO17 = (
    0,
    15,
    14,
    17,
    16,
    5,
    2,
    6,
    3,
    7,
    4,
    11,
    8,
    12,
    9,
    13,
    10,
)


COCO17_BONE_PAIRS = [
    (0, 0),
    (1, 0), (2, 0),
    (3, 1), (4, 2),
    (5, 6), (6, 5),
    (7, 5), (8, 6),
    (9, 7), (10, 8),
    (11, 12), (12, 11),
    (13, 11), (14, 12),
    (15, 13), (16, 14),
]


def validate_openpose18_joint_shape(joint_data):
    if joint_data.ndim != 5:
        raise ValueError('Expected OpenPose18 joint data with 5 dims (N, C, T, 18, M), got {}.'.format(joint_data.shape))

    n, c, t, v, m = joint_data.shape
    if c < 2:
        raise ValueError('Expected at least 2 channels for joint data, but got C={}.'.format(c))
    if v != 18:
        raise ValueError('Expected OpenPose18 joint data with V=18, but got V={}.'.format(v))

    return n, c, t, v, m


def validate_coco17_joint_shape(joint_data):
    if joint_data.ndim != 5:
        raise ValueError('Expected COCO17 joint data with 5 dims (N, C, T, 17, M), got {}.'.format(joint_data.shape))

    n, c, t, v, m = joint_data.shape
    if c < 2:
        raise ValueError('Expected at least 2 channels for joint data, but got C={}.'.format(c))
    if v != 17:
        raise ValueError('Expected COCO17 joint data with V=17, but got V={}.'.format(v))

    return n, c, t, v, m


def convert_openpose18_joint_to_coco17(openpose18_joint, coco17_joint=None):
    n, c, t, _, m = validate_openpose18_joint_shape(openpose18_joint)
    target_shape = (n, c, t, 17, m)

    if coco17_joint is None:
        coco17_joint = np.empty(target_shape, dtype=openpose18_joint.dtype)
    elif coco17_joint.shape != target_shape:
        raise ValueError('Expected coco17_joint shape {}, but got {}.'.format(target_shape, coco17_joint.shape))

    for coco_idx, op_idx in enumerate(OPENPOSE18_TO_COCO17):
        coco17_joint[:, :, :, coco_idx, :] = openpose18_joint[:, :, :, op_idx, :]

    return coco17_joint


def generate_coco17_bone_from_joint(coco17_joint, coco17_bone=None):
    target_shape = validate_coco17_joint_shape(coco17_joint)

    if coco17_bone is None:
        coco17_bone = np.zeros(target_shape, dtype=coco17_joint.dtype)
    elif coco17_bone.shape != target_shape:
        raise ValueError('Expected coco17_bone shape {}, but got {}.'.format(target_shape, coco17_bone.shape))
    else:
        coco17_bone[...] = 0

    for child, parent in COCO17_BONE_PAIRS:
        coco17_bone[:, :, :, child, :] = coco17_joint[:, :, :, child, :] - coco17_joint[:, :, :, parent, :]

    return coco17_bone


def prepare_output_path(path_str, overwrite=False):
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError('Output file already exists: {}. Use --overwrite to replace it.'.format(path))
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OpenPose18 joint data to COCO17 joint/bone data. Neck is dropped because COCO17 has no neck joint.'
    )
    parser.add_argument('--joint-in', required=True, help='Path to OpenPose18 joint .npy file with shape (N, C, T, 18, M).')
    parser.add_argument('--joint-out', required=True, help='Path to output COCO17 joint .npy file with shape (N, C, T, 17, M).')
    parser.add_argument('--bone-in', default=None, help='Optional OpenPose18 bone .npy file. It is accepted for compatibility but not used.')
    parser.add_argument('--bone-out', default=None, help='Optional output COCO17 bone .npy file. If set, bone data is regenerated from converted joints.')
    parser.add_argument('--mmap', dest='use_mmap', action='store_true', default=True,
                        help="Load the input joint file with mmap_mode='r' (default).")
    parser.add_argument('--no-mmap', dest='use_mmap', action='store_false',
                        help='Disable memory-mapped loading for the input joint file.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files.')
    return parser.parse_args()


def main():
    args = parse_args()

    joint_in_path = Path(args.joint_in)
    if not joint_in_path.is_file():
        raise FileNotFoundError('Input joint file not found: {}'.format(joint_in_path))

    joint_out_path = prepare_output_path(args.joint_out, overwrite=args.overwrite)
    bone_out_path = prepare_output_path(args.bone_out, overwrite=args.overwrite) if args.bone_out else None

    if args.bone_in:
        print('Ignoring --bone-in={} and regenerating COCO17 bone data from converted joints.'.format(args.bone_in))

    joint_in = np.load(str(joint_in_path), mmap_mode='r' if args.use_mmap else None)
    n, c, t, _, m = validate_openpose18_joint_shape(joint_in)

    joint_out = open_memmap(
        str(joint_out_path),
        mode='w+',
        dtype=joint_in.dtype,
        shape=(n, c, t, 17, m),
    )
    convert_openpose18_joint_to_coco17(joint_in, joint_out)
    if hasattr(joint_out, 'flush'):
        joint_out.flush()

    bone_out = None
    if bone_out_path is not None:
        bone_out = open_memmap(
            str(bone_out_path),
            mode='w+',
            dtype=joint_in.dtype,
            shape=(n, c, t, 17, m),
        )
        generate_coco17_bone_from_joint(joint_out, bone_out)
        if hasattr(bone_out, 'flush'):
            bone_out.flush()

    print('Input joint shape: {}'.format(tuple(joint_in.shape)))
    print('Output joint shape: {}'.format(tuple(joint_out.shape)))
    print('Joint output: {}'.format(joint_out_path))
    if bone_out is not None:
        print('Output bone shape: {}'.format(tuple(bone_out.shape)))
        print('Bone output: {}'.format(bone_out_path))


if __name__ == '__main__':
    main()
