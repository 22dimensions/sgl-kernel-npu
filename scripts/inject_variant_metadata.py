#!/usr/bin/env python3
"""
Inject WheelNext (PEP 817) variant.json into a built wheel and rename it.

Usage:
  python inject_variant_metadata.py --wheel PKG.whl \\
      --cann-version 8.5.0 --npu-hardware a3 --torch-version 2.8.0

The original wheel is replaced by a variant-labeled copy, e.g.:
  sgl_kernel_npu-2026.3.1-cp311-cp311-linux_aarch64.whl
  → sgl_kernel_npu-2026.3.1-cp311-cp311-linux_aarch64-850_a3_280.whl
"""
import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path

SCHEMA_URL = "https://variants-schema.wheelnext.dev/v0.0.3.json"
NAMESPACE = "cann_npu"

# Priority-ordered lists: most-preferred first
_ALL_CANN = ["8.5.0", "8.3.rc2"]
_ALL_NPU = ["a3", "910b"]
_ALL_TORCH = ["2.10.0", "2.8.0"]

# Variant label: max 16 chars, regex ^[0-9a-z._]{1,16}$
# Char budgets: cann≤8, npu≤4, torch≤6, two separators = ≤20 → trim cann to 5
_LABEL_RE = re.compile(r"^[0-9a-z._]{1,16}$")


def _slug(s: str, max_len: int) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())[:max_len]


def make_label(cann: str, npu: str, torch_ver: str, custom: str | None = None) -> str:
    if custom:
        assert _LABEL_RE.match(custom), f"Invalid variant label: {custom!r}"
        return custom
    # e.g. "8.5.0"→"850", "8.3.rc2"→"83rc2", "2.10.0"→"2100"
    label = f"{_slug(cann, 5)}_{_slug(npu, 4)}_{_slug(torch_ver, 6)}"
    assert _LABEL_RE.match(label), f"Generated label exceeds 16 chars or has bad chars: {label!r}"
    return label


def make_variant_json(cann: str, npu: str, torch_ver: str, label: str) -> dict:
    return {
        "$schema": SCHEMA_URL,
        "default-priorities": {
            "namespace": [NAMESPACE],
            "feature": {NAMESPACE: ["cann_version", "npu_hardware", "torch_version"]},
            "property": {
                NAMESPACE: {
                    "cann_version": _ALL_CANN,
                    "npu_hardware": _ALL_NPU,
                    "torch_version": _ALL_TORCH,
                }
            },
        },
        "providers": {
            NAMESPACE: {
                "install-time": True,
                "requires": ["cann-npu-variant-provider>=0.1"],
                "plugin-api": "cann_npu_variant_provider.plugin:CannNpuVariantProvider",
            }
        },
        "variants": {
            label: {
                NAMESPACE: {
                    "cann_version": [cann],
                    "npu_hardware": [npu],
                    "torch_version": [torch_ver],
                }
            }
        },
    }


def _find_dist_info(names: list[str]) -> str:
    candidates = {n.split("/")[0] for n in names if ".dist-info" in n}
    assert candidates, "No .dist-info directory found in wheel"
    return next(iter(candidates))


def inject(
    wheel_path: str,
    cann: str,
    npu: str,
    torch_ver: str,
    custom_label: str | None = None,
) -> Path:
    src = Path(wheel_path)
    assert src.exists(), f"Wheel not found: {src}"

    label = make_label(cann, npu, torch_ver, custom_label)
    variant_content = json.dumps(make_variant_json(cann, npu, torch_ver, label), indent=2)

    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()

    dist_info_dir = _find_dist_info(names)
    variant_entry = f"{dist_info_dir}/variant.json"

    # New wheel filename: append variant label before .whl
    dst = src.parent / f"{src.stem}-{label}.whl"
    tmp = src.with_suffix(".tmp.zip")

    with zipfile.ZipFile(src, "r") as r, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as w:
        for info in r.infolist():
            w.writestr(info, r.read(info.filename))
        w.writestr(variant_entry, variant_content)

    shutil.move(tmp, dst)
    src.unlink()  # remove original unlabeled wheel
    print(f"Injected variant metadata → {dst.name}")
    return dst


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wheel", required=True, help="Path to the built .whl file")
    p.add_argument("--cann-version", required=True, help="CANN toolkit version, e.g. 8.5.0")
    p.add_argument("--npu-hardware", required=True, help="NPU chip generation: a3 or 910b")
    p.add_argument("--torch-version", required=True, help="PyTorch version, e.g. 2.8.0")
    p.add_argument("--variant-label", default=None, help="Override auto-generated variant label (max 16 chars)")
    args = p.parse_args()
    inject(args.wheel, args.cann_version, args.npu_hardware, args.torch_version, args.variant_label)


if __name__ == "__main__":
    main()
