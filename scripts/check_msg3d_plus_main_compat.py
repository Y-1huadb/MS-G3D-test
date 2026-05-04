#!/usr/bin/env python3
"""Check main.py compatibility after ONNX-safe model refactor."""

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import import_class  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check msg3d_plus checkpoint/main.py compatibility")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--weights", type=str, required=True, help="Path to checkpoint/weights")
    parser.add_argument("--T", type=int, default=300, help="Temporal length for dummy input")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config should parse to dict, got {type(cfg)}")
    return cfg


def is_tensor_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if is_tensor_dict(ckpt):
        return ckpt, "checkpoint(root)"

    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in ckpt:
                val = ckpt[key]
                if is_tensor_dict(val):
                    return val, f"checkpoint['{key}']"
                if isinstance(val, torch.nn.Module):
                    return val.state_dict(), f"checkpoint['{key}'].state_dict()"

        filtered = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if len(filtered) > 0:
            return filtered, "checkpoint(filtered_tensor_items)"

    raise RuntimeError(
        "Unable to locate state_dict in checkpoint. Supported: raw state_dict or keys "
        "state_dict/model/model_state_dict/net."
    )


def strip_known_prefixes(key: str) -> str:
    out = key
    prefixes = ("module.", "model.", "base_model.")
    while True:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p):]
                changed = True
        if not changed:
            break
    return out


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    normalized = OrderedDict()
    for k, v in state_dict.items():
        normalized[strip_known_prefixes(k)] = v
    return normalized


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    model_class_path = cfg.get("model")
    if not model_class_path:
        raise ValueError(f"Config missing 'model': {args.config}")
    model_args = cfg.get("model_args", {}) or {}
    if not isinstance(model_args, dict):
        raise ValueError("Config field 'model_args' must be a dict")

    in_channels = int(model_args.get("in_channels", 3))
    num_point = int(model_args.get("num_point", 18))
    num_person = int(model_args.get("num_person", 2))
    num_class = int(model_args.get("num_class", 400))

    device = torch.device(args.device)

    Model = import_class(model_class_path)
    model = Model(**model_args).to(device)
    model.eval()

    ckpt = torch.load(args.weights, map_location=device)
    raw_state_dict, source = extract_state_dict(ckpt)
    state_dict = normalize_state_dict_keys(raw_state_dict)

    incompat = model.load_state_dict(state_dict, strict=True)
    print(f"state_dict source: {source}")
    print(f"missing keys: {len(incompat.missing_keys)}")
    print(f"unexpected keys: {len(incompat.unexpected_keys)}")

    if len(incompat.missing_keys) != 0 or len(incompat.unexpected_keys) != 0:
        raise RuntimeError(
            f"Checkpoint incompatible: missing={len(incompat.missing_keys)}, "
            f"unexpected={len(incompat.unexpected_keys)}"
        )

    print("[PASS] checkpoint load compatible")

    x = torch.randn(1, in_channels, args.T, num_point, num_person, device=device, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    if isinstance(y, (tuple, list)):
        y = y[0]

    expected_shape = [1, num_class]
    actual_shape = list(y.shape)
    print(f"forward output shape: {actual_shape}")
    if actual_shape != expected_shape:
        raise RuntimeError(f"Output shape mismatch: expected {expected_shape}, got {actual_shape}")

    print("[PASS] forward compatible with main.py model path")


if __name__ == "__main__":
    main()
