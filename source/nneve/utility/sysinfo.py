import csv
import datetime
import platform as pyplatform
import subprocess
import time
from contextlib import suppress
from typing import Any, Dict, List, Tuple

import psutil
import tensorflow as tf
from cpuinfo import get_cpu_info
from pydantic import BaseModel, Field

__all__ = ["get_sys_info"]


class JsonPrintableModel(BaseModel):
    def __str__(self) -> str:
        return self.json(indent=2)


class PlatformInfo(JsonPrintableModel):
    architecture: Tuple[str, str] = Field(
        default_factory=pyplatform.architecture
    )
    machine: str = Field(default_factory=pyplatform.machine)
    node: str = Field(default_factory=pyplatform.node)
    platform: str = Field(default_factory=pyplatform.platform)
    processor: str = Field(default_factory=pyplatform.processor)
    release: str = Field(default_factory=pyplatform.release)
    system: str = Field(default_factory=pyplatform.system)
    version: str = Field(default_factory=pyplatform.version)


class PythonInfo(JsonPrintableModel):
    python_build: Tuple[str, str] = Field(
        default_factory=pyplatform.python_build
    )
    python_compiler: str = Field(default_factory=pyplatform.python_compiler)
    python_branch: str = Field(default_factory=pyplatform.python_branch)
    python_implementation: str = Field(
        default_factory=pyplatform.python_implementation
    )
    python_revision: str = Field(default_factory=pyplatform.python_revision)
    python_version: str = Field(default_factory=pyplatform.python_version)


class CpuInfo(JsonPrintableModel):
    brand_raw: str
    core_count: int
    architecture: str
    max_frequency: int

    L1_data: str
    L1_instruction: str
    L2_size: str
    L2_line_size: str
    L2_cache_associativity: str
    L3_size: str

    support_mmx: bool
    support_sse: bool
    support_sse2: bool
    support_ssse3: bool
    support_sse4_1: bool
    support_sse4_2: bool
    support_avx: bool
    support_avx2: bool
    support_avx512: bool

    stepping: str
    model: str
    family: str
    vendor_id_raw: str
    flags: str

    def __init__(self) -> None:
        kwargs: Dict[str, Any] = {}
        cpuinfo = get_cpu_info()
        kwargs.setdefault("core_count", cpuinfo.get("count", "N/A"))
        kwargs.setdefault("architecture", cpuinfo.get("arch", "N/A"))
        kwargs.setdefault(
            "max_frequency", cpuinfo.get("hz_advertised", ("N/A", "N/A"))[0]
        )
        kwargs.setdefault(
            "min_frequency", cpuinfo.get("hz_advertised", ("N/A", "N/A"))[1]
        )
        kwargs.setdefault("L1_data", cpuinfo.get("l1_data_cache_size", "N/A"))
        kwargs.setdefault(
            "L1_instruction", cpuinfo.get("l1_instruction_cache_size", "N/A")
        )
        kwargs.setdefault("L2_size", cpuinfo.get("l2_cache_size", "N/A"))
        kwargs.setdefault(
            "L2_line_size", cpuinfo.get("l2_cache_line_size", "N/A")
        )
        kwargs.setdefault(
            "L2_cache_associativity",
            cpuinfo.get("l2_cache_associativity", "N/A"),
        )
        kwargs.setdefault("L3_size", cpuinfo.get("l3_cache_size", "N/A"))
        kwargs.setdefault("stepping", cpuinfo.get("stepping", "N/A"))
        kwargs.setdefault("model", cpuinfo.get("model", "N/A"))
        kwargs.setdefault("family", cpuinfo.get("family", "N/A"))
        kwargs.setdefault("flags", ";".join(cpuinfo.get("flags", "N/A")))
        kwargs.setdefault("vendor_id_raw", cpuinfo.get("vendor_id_raw", "N/A"))
        kwargs.setdefault("brand_raw", cpuinfo.get("brand_raw", "N/A"))

        kwargs.setdefault("support_mmx", "mmx" in cpuinfo.get("flags", "N/A"))
        kwargs.setdefault("support_sse", "sse" in cpuinfo.get("flags", "N/A"))
        kwargs.setdefault(
            "support_sse2", "sse2" in cpuinfo.get("flags", "N/A")
        )
        kwargs.setdefault(
            "support_ssse3", "ssse3" in cpuinfo.get("flags", "N/A")
        )
        kwargs.setdefault(
            "support_sse4_1", "sse4_1" in cpuinfo.get("flags", "N/A")
        )
        kwargs.setdefault(
            "support_sse4_2", "sse4_2" in cpuinfo.get("flags", "N/A")
        )
        kwargs.setdefault("support_avx", "avx" in cpuinfo.get("flags", "N/A"))
        kwargs.setdefault(
            "support_avx2", "avx2" in cpuinfo.get("flags", "N/A")
        )
        kwargs.setdefault(
            "support_avx512", "avx512" in cpuinfo.get("flags", "N/A")
        )

        super().__init__(**kwargs)


class SysMemInfo(JsonPrintableModel):
    total: int = Field(default=-1)
    available: int = Field(default=-1)
    used: int = Field(default=-1)
    free: int = Field(default=-1)

    swap_total: int = Field(default=-1)
    swap_used: int = Field(default=-1)
    swap_free: int = Field(default=-1)

    def __init__(self) -> None:
        kwargs: Dict[str, Any] = {}
        with suppress(Exception):
            virtual = psutil.virtual_memory()  # type: ignore
            kwargs.setdefault("total", virtual.total)  # type: ignore
            kwargs.setdefault("available", virtual.available)  # type: ignore
            kwargs.setdefault("used", virtual.used)  # type: ignore
            kwargs.setdefault("free", virtual.free)  # type: ignore

        with suppress(Exception):
            swap = psutil.swap_memory()
            kwargs.setdefault("swap_total", swap.total)
            kwargs.setdefault("swap_used", swap.used)
            kwargs.setdefault("swap_free", swap.free)

        super().__init__(**kwargs)


class TfInfo(JsonPrintableModel):
    version: str = Field(default=tf.__version__)
    cpu_count: int = Field(
        default_factory=lambda: len(tf.config.get_visible_devices("CPU"))  # type: ignore
    )
    gpu_count: int = Field(
        default_factory=lambda: len(tf.config.get_visible_devices("GPU"))  # type: ignore
    )
    is_built_with_cuda: bool = Field(
        default_factory=tf.test.is_built_with_cuda
    )
    is_built_with_gpu_support: bool = Field(
        default_factory=tf.test.is_built_with_gpu_support
    )
    gpu_device_name: bool = Field(default_factory=tf.test.gpu_device_name)


class GpuInfo(JsonPrintableModel):
    index: str
    name: str
    uuid: str
    count: str
    temperature_gpu: str

    vram_total: str
    vram_free: str
    vram_used: str
    temperature_vram: str
    ecc_vram: str

    driver_version: str
    vbios_version: str

    pci_bus_id: str
    pcie_link_gen_max: str
    pcie_link_gen_current: str

    clocks_vram: str
    clocks_gpu: str

    power_max: str
    compute_cap: str

    @classmethod
    def get_gpu_info(cls) -> List["GpuInfo"]:
        gpuinfo: List[GpuInfo] = []
        with suppress(Exception):
            gpuinfo.extend(cls.get_nvidia_gpu_info())
        with suppress(Exception):
            gpuinfo.extend(cls.get_amd_gpu_info())
        with suppress(Exception):
            gpuinfo.extend(cls.get_intel_gpu_info())
        return gpuinfo

    @classmethod
    def get_nvidia_gpu_info(cls) -> List["GpuInfo"]:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu="
                    "index,count,gpu_name,uuid,temperature.gpu,"
                    "memory.total,memory.free,memory.used,temperature.memory,ecc.mode.current,"
                    "driver_version,vbios_version,"
                    "pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,"
                    "clocks.max.memory,clocks.max.graphics,"
                    "power.limit,compute_cap"
                ),
                "--format=csv",
            ]
        )
        reader = csv.DictReader(output.decode("utf-8").split("\n"))
        gpuinfo: List[GpuInfo] = []
        for gpu in reader:
            gpuinfo.append(
                cls(
                    index=gpu["index"].strip(),
                    name=gpu[" name"].strip(),
                    uuid=gpu[" uuid"].strip(),
                    count=gpu[" count"].strip(),
                    temperature_gpu=gpu[" temperature.gpu"].strip(),
                    vram_total=gpu[" memory.total [MiB]"].strip(),
                    vram_free=gpu[" memory.free [MiB]"].strip(),
                    vram_used=gpu[" memory.used [MiB]"].strip(),
                    temperature_vram=gpu[" temperature.memory"].strip(),
                    ecc_vram=gpu[" ecc.mode.current"].strip(),
                    driver_version=gpu[" driver_version"].strip(),
                    vbios_version=gpu[" vbios_version"].strip(),
                    pci_bus_id=gpu[" pci.bus_id"].strip(),
                    pcie_link_gen_max=gpu[" pcie.link.gen.max"].strip(),
                    pcie_link_gen_current=gpu[
                        " pcie.link.gen.current"
                    ].strip(),
                    clocks_vram=gpu[" clocks.max.memory [MHz]"].strip(),
                    clocks_gpu=gpu[" clocks.max.graphics [MHz]"].strip(),
                    power_max=gpu[" power.limit [W]"].strip(),
                    compute_cap=gpu[" compute_cap"].strip(),
                )
            )
        return gpuinfo

    @classmethod
    def get_amd_gpu_info(cls) -> List["GpuInfo"]:
        return []

    @classmethod
    def get_intel_gpu_info(cls) -> List["GpuInfo"]:
        return []


class SysInfo(JsonPrintableModel):

    creation_time: datetime.datetime = Field(
        default_factory=datetime.datetime.now
    )
    timezone: str = Field(default=time.tzname[1])
    platform: PlatformInfo = Field(default_factory=PlatformInfo)
    python: PythonInfo = Field(default_factory=PythonInfo)
    cpu: CpuInfo = Field(default_factory=CpuInfo)
    memory: SysMemInfo = Field(default_factory=SysMemInfo)
    tensorflow: TfInfo = Field(default_factory=TfInfo)
    gpu: List[GpuInfo] = Field(default_factory=GpuInfo.get_gpu_info)


def get_sys_info() -> SysInfo:
    return SysInfo()
