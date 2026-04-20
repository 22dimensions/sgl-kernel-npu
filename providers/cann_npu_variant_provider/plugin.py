import glob
import os
import re
import subprocess
from dataclasses import dataclass, field


NAMESPACE = "cann_npu"

# Priority-ordered lists: most-preferred first
_ALL_CANN    = ["8.5.0", "8.3.rc2"]
_ALL_NPU     = ["a3", "910b"]
_ALL_TORCH   = ["2.10.0", "2.8.0"]


@dataclass
class VariantFeatureConfig:
    name: str
    values: list
    multi_value: bool = False


class CannNpuVariantProvider:
    """WheelNext variant provider for Ascend CANN and NPU hardware."""

    namespace = NAMESPACE
    is_aot_plugin = False  # requires runtime detection

    @classmethod
    def get_all_configs(cls) -> list:
        return [
            VariantFeatureConfig("cann_version", _ALL_CANN),
            VariantFeatureConfig("npu_hardware", _ALL_NPU),
            VariantFeatureConfig("torch_version", _ALL_TORCH),
        ]

    @classmethod
    def get_supported_configs(cls) -> list:
        configs = []

        cann = cls._detect_cann_version()
        if cann and cann in _ALL_CANN:
            # Accept wheels built for same or older CANN
            idx = _ALL_CANN.index(cann)
            configs.append(VariantFeatureConfig("cann_version", _ALL_CANN[idx:]))

        npu = cls._detect_npu_hardware()
        if npu:
            configs.append(VariantFeatureConfig("npu_hardware", [npu]))

        torch_ver = cls._detect_torch_version()
        if torch_ver and torch_ver in _ALL_TORCH:
            configs.append(VariantFeatureConfig("torch_version", [torch_ver]))

        return configs

    # --- detection helpers ---

    @classmethod
    def _detect_cann_version(cls) -> str | None:
        if override := os.environ.get("CANN_VERSION"):
            return override
        info_file = "/etc/Ascend/ascend_cann_install.info"
        if os.path.exists(info_file):
            m = re.search(r"Version=([\d.rc]+)", open(info_file).read())
            if m:
                return m.group(1)
        return None

    @classmethod
    def _detect_npu_hardware(cls) -> str | None:
        if override := os.environ.get("NPU_HARDWARE"):
            return override
        if not glob.glob("/dev/davinci*"):
            return None
        try:
            out = subprocess.check_output(["npu-smi", "info"], text=True, timeout=5)
            if re.search(r"Ascend\s*A3|A3\b", out, re.I):
                return "a3"
            if re.search(r"Ascend\s*910B|910B\b", out, re.I):
                return "910b"
        except Exception:
            pass
        return None

    @classmethod
    def _detect_torch_version(cls) -> str | None:
        try:
            import torch
            raw = torch.__version__.split("+")[0]
            return ".".join(raw.split(".")[:3])
        except ImportError:
            return None
