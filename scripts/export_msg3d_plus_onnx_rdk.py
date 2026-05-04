#!/usr/bin/env python3
"""Export MSG3D / MSG3D-Plus model to fixed-shape 4D ONNX for RDK X5.

External ONNX input layout:
    [N, M*C, T, V]

Internal model input layout:
    [N, C, T, V, M]
"""

import argparse
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import import_class  # noqa: E402


class RDKMSG3DWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, in_channels: int = 3, num_person: int = 2):
        super().__init__()
        self.base_model = base_model
        self.in_channels = int(in_channels)
        self.num_person = int(num_person)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, M*C, T, V]
        N, MC, T, V = x.shape
        C = self.in_channels
        M = self.num_person

        assert MC == M * C, f"Expected channel={M * C}, got {MC}"

        # channel order:
        # p0_x, p0_y, p0_conf, p1_x, p1_y, p1_conf
        x = x.view(N, M, C, T, V)

        # restore to original model input:
        # [N, C, T, V, M]
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        return self.base_model(x)


def msg3d_5d_to_rdk_4d(x5: torch.Tensor) -> torch.Tensor:
    """
    x5: [N, C, T, V, M]
    return x4: [N, M*C, T, V]
    channel order:
        p0_x, p0_y, p0_conf, p1_x, p1_y, p1_conf
    """
    if x5.dim() != 5:
        raise ValueError(f"Expected 5D tensor [N,C,T,V,M], got shape={tuple(x5.shape)}")

    N, C, T, V, M = x5.shape
    x4 = x5.permute(0, 4, 1, 2, 3).contiguous()
    x4 = x4.view(N, M * C, T, V)
    return x4


def rdk_4d_to_msg3d_5d(x4: torch.Tensor, in_channels: int = 3, num_person: int = 2) -> torch.Tensor:
    """
    x4: [N, M*C, T, V]
    return x5: [N, C, T, V, M]
    """
    if x4.dim() != 4:
        raise ValueError(f"Expected 4D tensor [N,M*C,T,V], got shape={tuple(x4.shape)}")

    N, MC, T, V = x4.shape
    C = int(in_channels)
    M = int(num_person)

    assert MC == M * C, f"Expected channel={M * C}, got {MC}"

    x5 = x4.view(N, M, C, T, V)
    x5 = x5.permute(0, 2, 3, 4, 1).contiguous()
    return x5


def test_layout_conversion(T: int = 300, V: int = 18, C: int = 3, M: int = 2) -> None:
    x5 = torch.arange(1 * C * T * V * M, dtype=torch.float32).view(1, C, T, V, M)
    x4 = msg3d_5d_to_rdk_4d(x5)
    x5_recovered = rdk_4d_to_msg3d_5d(x4, in_channels=C, num_person=M)

    if not torch.equal(x5, x5_recovered):
        raise AssertionError("Layout conversion mismatch: x5 != recovered_x5")

    print(f"x4 shape: {list(x4.shape)}")
    print("[PASS] layout conversion test passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MSG3D/MSG3D-Plus to fixed-shape ONNX [N, M*C, T, V] for RDK X5"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/kinetics-skeleton/train_joint_dynamic_part.yaml",
        help="Path to YAML config used in training.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Model checkpoint path. Required unless --test-layout is set.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="msg3d_plus_dynamic_part_t300_rdk.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument("--T", type=int, default=300, help="Temporal length T.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size N for dummy input.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu / cuda:0")
    parser.add_argument("--check", action="store_true", help="Run onnxruntime numerical check after export.")
    parser.add_argument("--test-layout", action="store_true", help="Only run layout conversion test.")
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must parse to dict, got: {type(cfg)}")
    return cfg


def is_tensor_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and all(torch.is_tensor(v) for v in obj.values())


def extract_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if is_tensor_dict(ckpt):
        return ckpt, "checkpoint(root)"

    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net"):
            if k in ckpt:
                val = ckpt[k]
                if is_tensor_dict(val):
                    return val, f"checkpoint['{k}']"
                if isinstance(val, nn.Module):
                    return val.state_dict(), f"checkpoint['{k}'].state_dict()"

        # Fallback: pick tensor entries from root dict.
        filtered = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if len(filtered) > 0:
            return filtered, "checkpoint(filtered_tensor_items)"

    raise RuntimeError(
        "Unable to locate state_dict in checkpoint. Supported keys: "
        "state_dict / model / model_state_dict / net, or a raw state_dict file."
    )


def strip_known_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "base_model.")
    out = key
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


def print_key_list(title: str, keys, limit: int = 20) -> None:
    keys = list(keys)
    print(f"{title}: {len(keys)}")
    if not keys:
        return
    for k in keys[:limit]:
        print(f"  - {k}")
    if len(keys) > limit:
        print(f"  ... ({len(keys) - limit} more)")


def load_model_weights(model: nn.Module, weights_path: str, device: torch.device) -> None:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device)
    raw_state_dict, source = extract_state_dict(ckpt)
    state_dict = normalize_state_dict_keys(raw_state_dict)

    incompat = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded weights from: {weights_path}")
    print(f"state_dict source: {source}")
    print_key_list("missing keys", incompat.missing_keys)
    print_key_list("unexpected keys", incompat.unexpected_keys)


def run_onnx_check(onnx_path: str, wrapper: nn.Module, dummy: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for --check. Install with: pip install onnxruntime") from exc

    wrapper.eval()
    with torch.no_grad():
        torch_out = wrapper(dummy)
    if isinstance(torch_out, (tuple, list)):
        torch_out = torch_out[0]

    dummy_np = dummy.detach().cpu().numpy()
    torch_np = torch_out.detach().cpu().numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    onnx_out = ort_session.run(None, {input_name: dummy_np})[0]

    diff = np.abs(torch_np - onnx_out)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    print(f"pytorch output shape: {list(torch_np.shape)}")
    print(f"onnx output shape: {list(onnx_out.shape)}")
    print(f"max_abs_diff: {max_abs_diff:.8f}")
    print(f"mean_abs_diff: {mean_abs_diff:.8f}")

    if max_abs_diff < 1e-4:
        print("[PASS] ONNX numerical check passed.")
    else:
        print("[WARNING] ONNX numerical difference is larger than expected.")


def validate_exported_onnx(onnx_path: str, expected_opset: int = 11) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError("onnx is required to validate exported model. Install with: pip install onnx") from exc

    model = onnx.load(onnx_path)
    opset_by_domain = {item.domain: int(item.version) for item in model.opset_import}
    ai_onnx_opset = opset_by_domain.get("", None)
    has_einsum = any(node.op_type == "Einsum" for node in model.graph.node)

    print(f"onnx ai.onnx opset: {ai_onnx_opset}")
    print(f"contains Einsum node: {has_einsum}")

    if ai_onnx_opset != expected_opset:
        raise RuntimeError(f"Expected ai.onnx opset {expected_opset}, got {ai_onnx_opset}")
    if has_einsum:
        raise RuntimeError("Exported ONNX graph contains Einsum node, incompatible with opset11 target flow.")

    print("[PASS] ONNX graph check passed (opset and node type constraints).")


def main() -> None:
    args = parse_args()

    if args.test_layout:
        C = 3
        M = 2
        V = 18
        if os.path.exists(args.config):
            cfg_for_layout = load_yaml_config(args.config)
            model_args_for_layout = cfg_for_layout.get("model_args", {}) or {}
            C = int(model_args_for_layout.get("in_channels", 3))
            M = int(model_args_for_layout.get("num_person", 2))
            V = int(model_args_for_layout.get("num_point", 18))
        test_layout_conversion(T=args.T, V=V, C=C, M=M)
        return

    if not args.weights:
        raise ValueError("--weights is required unless --test-layout is set.")

    cfg = load_yaml_config(args.config)

    model_class_path = cfg.get("model")
    if not model_class_path:
        raise ValueError(f"Config missing 'model': {args.config}")

    model_args = cfg.get("model_args", {}) or {}
    if not isinstance(model_args, dict):
        raise ValueError("Config 'model_args' must be a dict.")

    in_channels = int(model_args.get("in_channels", 3))
    num_person = int(model_args.get("num_person", 2))
    num_point = int(model_args.get("num_point", 18))
    num_class = int(model_args.get("num_class", 400))

    device = torch.device(args.device)

    Model = import_class(model_class_path)
    base_model = Model(**model_args)
    base_model = base_model.to(device)

    load_model_weights(base_model, args.weights, device)

    base_model.eval()
    wrapper = RDKMSG3DWrapper(base_model, in_channels=in_channels, num_person=num_person).to(device)
    wrapper.eval()

    dummy = torch.randn(
        args.batch_size,
        in_channels * num_person,
        args.T,
        num_point,
        dtype=torch.float32,
        device=device,
    )

    # Export pre-check: run one forward first to ensure graph is valid in eval mode.
    with torch.no_grad():
        logits = wrapper(dummy)

    if isinstance(logits, (tuple, list)):
        logits_for_shape = logits[0]
    else:
        logits_for_shape = logits

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=None,
            do_constant_folding=True,
            opset_version=args.opset,
            dynamo=False,
            external_data=False,
        )

    validate_exported_onnx(args.output, expected_opset=args.opset)

    print("Model class from config:")
    print(f"    {model_class_path}")
    print("")
    print("Original model input shape:")
    print(f"    [N, C, T, V, M] = [{args.batch_size}, {in_channels}, {args.T}, {num_point}, {num_person}]")
    print("")
    print("RDK ONNX external input shape:")
    print(f"    [N, M*C, T, V] = [{args.batch_size}, {in_channels * num_person}, {args.T}, {num_point}]")
    print("")
    print("Channel order:")
    print("    p0_x, p0_y, p0_conf, p1_x, p1_y, p1_conf")
    print("")
    print("Output shape:")
    print(f"    [N, num_class] = [{args.batch_size}, {num_class}]")
    print(f"    (actual forward output shape: {list(logits_for_shape.shape)})")
    print("")
    print("ONNX saved to:")
    print(f"    {args.output}")

    if args.check:
        run_onnx_check(args.output, wrapper, dummy)


if __name__ == "__main__":
    main()
