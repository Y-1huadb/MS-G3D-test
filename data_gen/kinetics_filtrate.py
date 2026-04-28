import argparse
import os
import pickle
from collections import defaultdict, Counter

import numpy as np


def load_label_pkl(label_path):
    with open(label_path, "rb") as f:
        obj = pickle.load(f)

    # MS-G3D / ST-GCN 常见格式: (sample_name, label)
    if isinstance(obj, tuple) and len(obj) == 2:
        sample_name, label = obj
        return list(sample_name), list(label)

    # 兼容少数 dict 格式
    if isinstance(obj, dict):
        if "sample_name" in obj and "label" in obj:
            return list(obj["sample_name"]), list(obj["label"])

    raise ValueError(f"Unsupported label pkl format: {type(obj)}")


def save_label_pkl(out_path, sample_name, label):
    with open(out_path, "wb") as f:
        pickle.dump((sample_name, label), f)


def make_subset(
    data_path,
    label_path,
    out_data_path,
    out_label_path,
    max_per_class=500,
    seed=2026,
    chunk_size=256,
):
    os.makedirs(os.path.dirname(out_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_label_path), exist_ok=True)

    print(f"Loading labels: {label_path}")
    sample_name, label = load_label_pkl(label_path)

    print(f"Loading data header: {data_path}")
    data = np.load(data_path, mmap_mode="r")

    n = data.shape[0]
    assert len(sample_name) == n, f"sample_name length {len(sample_name)} != data N {n}"
    assert len(label) == n, f"label length {len(label)} != data N {n}"

    print(f"Original data shape: {data.shape}")
    print(f"Original num samples: {n}")
    print(f"Original num classes: {len(set(label))}")

    label_to_indices = defaultdict(list)
    for i, y in enumerate(label):
        label_to_indices[y].append(i)

    rng = np.random.default_rng(seed)

    selected_indices = []
    per_class_count = {}

    for y, indices in sorted(label_to_indices.items(), key=lambda x: x[0]):
        indices = np.array(indices, dtype=np.int64)

        if len(indices) > max_per_class:
            chosen = rng.choice(indices, size=max_per_class, replace=False)
        else:
            chosen = indices

        chosen = sorted(chosen.tolist())
        selected_indices.extend(chosen)
        per_class_count[y] = len(chosen)

    # 保持原始数据顺序，避免打乱 label 和 data 的对应关系
    selected_indices = sorted(selected_indices)

    new_sample_name = [sample_name[i] for i in selected_indices]
    new_label = [label[i] for i in selected_indices]

    print(f"Selected num samples: {len(selected_indices)}")
    print(f"Selected num classes: {len(set(new_label))}")
    print(f"Max per class: {max(Counter(new_label).values())}")
    print(f"Min per class: {min(Counter(new_label).values())}")

    out_shape = (len(selected_indices),) + data.shape[1:]
    print(f"Output data shape: {out_shape}")

    print(f"Writing subset data to: {out_data_path}")
    out_data = np.lib.format.open_memmap(
        out_data_path,
        mode="w+",
        dtype=data.dtype,
        shape=out_shape,
    )

    selected_indices = np.array(selected_indices, dtype=np.int64)

    for start in range(0, len(selected_indices), chunk_size):
        end = min(start + chunk_size, len(selected_indices))
        idx = selected_indices[start:end]
        out_data[start:end] = data[idx]

        if start % (chunk_size * 20) == 0:
            print(f"Written {end}/{len(selected_indices)} samples")

    del out_data

    print(f"Writing subset labels to: {out_label_path}")
    save_label_pkl(out_label_path, new_sample_name, new_label)

    print("Done.")

    print("\nClass count preview:")
    counter = Counter(new_label)
    for y, c in list(sorted(counter.items(), key=lambda x: x[0]))[:20]:
        print(f"class {y}: {c}")

    if len(counter) > 20:
        print("...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="/lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/data/kinetics/train_data_joint.npy", help="Path to train_data.npy or val_data.npy")
    parser.add_argument("--label", default="/lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/data/kinetics/train_label.pkl", help="Path to train_label.pkl or val_label.pkl")
    parser.add_argument("--out-data", default="/lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/data/kinetics_500/train_data_joint.npy", help="Output npy path")
    parser.add_argument("--out-label", default="/lab/haoq_lab/cse12211219/action_classify/MS-G3D-test/data/kinetics_500/train_label.pkl", help="Output pkl path")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=256)

    args = parser.parse_args()

    make_subset(
        data_path=args.data,
        label_path=args.label,
        out_data_path=args.out_data,
        out_label_path=args.out_label,
        max_per_class=args.max_per_class,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )