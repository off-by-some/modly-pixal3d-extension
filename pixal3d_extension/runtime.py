from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import importlib
import importlib.machinery
import inspect
import faulthandler
import math
import os
import random
import re
import sys
import time
import traceback
import tempfile
import threading
import types
import uuid
from pathlib import Path
from typing import Any, Callable

from pixal3d_extension.assets import bootstrap_auxiliary_assets, normalize_auxiliary_source_mode, resolve_auxiliary_sources, requires_local_auxiliary
from pixal3d_extension.paths import derive_modly_home, is_contained_path, require_contained_path
from pixal3d_extension.pipeline_patch import patch_pipeline, validate_pipeline_patch

PIXAL3D_MODEL_SOURCE = "TencentARC/Pixal3D"
PIXAL3D_TEXTURE_SIZE_ENV = "PIXAL3D_TEXTURE_SIZE"
SUPPORTED_TEXTURE_SIZES = (1024, 2048)
DEFAULT_TEXTURE_SIZE = 1024
_TEXTURE_EXPORT_LOCK = threading.RLock()
SUPPORTED_MANUAL_FOVS = (-1.0, 0.2, 0.35, 0.5)
DEFAULT_MANUAL_FOV = -1.0
_NATTEN_CPU_SPILL_THRESHOLD_BYTES = 512 * 1024 * 1024
_NATTEN_CHUNK_TARGET_BYTES = 128 * 1024 * 1024
SUPPORTED_RUNTIME_LANES = {
    "linux-aarch64-cp312-cuda124",
    "linux-x64-cp312-cuda124",
    "windows-x64-cp311-cuda124",
    "windows-x64-cp312-cuda124",
}
EXPERIMENTAL_BASE_RUNTIME_LANES = {"windows-x64-cp311-cuda128-blackwell"}
SUPPORTED_RUNTIME_LANE = "linux-aarch64-cp312-cuda124"


def _enable_crash_diagnostics() -> None:
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        return


def _diagnostic_checkpoint(label: str) -> None:
    print(f"[Pixal3D Diagnostic] {label}", file=sys.stderr, flush=True)


def _default_pipeline_factory(source: str) -> Any:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    module = importlib.import_module("pixal3d.pipelines")
    return module.Pixal3DImageTo3DPipeline.from_pretrained(source)


def _prepare_runtime_compat() -> None:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "0")
    import torch

    if not hasattr(torch.nn.Module, "all_tied_weights_keys"):
        torch.nn.Module.all_tied_weights_keys = {}


def _install_windows_native_module_aliases() -> None:
    """Expose Windows exact-stack native wheels under upstream module names.

    The Windows wheels are published as alias packages (`cumesh_vb`,
    `flex_gemm_ap`, `o_voxel_vb_ap`) while Pixal3D upstream imports the Linux
    module names (`cumesh`, `flex_gemm`, `o_voxel`). Keep the installed package
    names exact-stack, but provide import aliases before importing upstream
    inference code.
    """

    if os.name != "nt":
        return

    aliases = {
        "cumesh": "cumesh_vb",
        "flex_gemm": "flex_gemm_ap",
        "o_voxel": "o_voxel_vb_ap",
    }
    for upstream_name, windows_name in aliases.items():
        if upstream_name in sys.modules:
            if upstream_name == "o_voxel":
                _install_windows_o_voxel_python_compat(sys.modules[upstream_name])
            continue
        try:
            module = importlib.import_module(windows_name)
            sys.modules[upstream_name] = module
            if upstream_name == "o_voxel":
                _install_windows_o_voxel_python_compat(module)
        except ModuleNotFoundError:
            continue


def _install_windows_o_voxel_python_compat(o_voxel_module: Any) -> None:
    """Expose Python modules omitted by the Windows o_voxel alias wheel.

    `o_voxel_vb_ap` ships the native `_C` extension and conversion helpers but
    omits upstream Python modules used by Pixal3D (`o_voxel.postprocess` and
    `o_voxel.rasterize`). Install package attributes and submodule aliases so
    upstream imports keep using `o_voxel.*` while native calls resolve to the
    Windows wheel.
    """

    if os.name != "nt":
        return

    if "o_voxel._C" not in sys.modules:
        try:
            sys.modules["o_voxel._C"] = importlib.import_module("o_voxel_vb_ap._C")
            setattr(o_voxel_module, "_C", sys.modules["o_voxel._C"])
        except ModuleNotFoundError:
            pass

    compat_modules = {
        "postprocess": "pixal3d_extension.o_voxel_compat.postprocess",
        "rasterize": "pixal3d_extension.o_voxel_compat.rasterize",
    }
    for attr_name, compat_name in compat_modules.items():
        full_name = f"o_voxel.{attr_name}"
        if full_name in sys.modules:
            setattr(o_voxel_module, attr_name, sys.modules[full_name])
            continue
        compat_module = importlib.import_module(compat_name)
        sys.modules[full_name] = compat_module
        setattr(o_voxel_module, attr_name, compat_module)


def _import_module_with_torch_compile_disabled(module_name: str) -> Any:
    """Import a module while suppressing torch.compile import-time side effects.

    NATTEN v0.21.x decorates some helpers with torch.compile at import time.
    On the Modly Windows cp311/cu124 lane this can enter torch._inductor and a
    Triton package whose public API does not expose AttrsDescriptor. The native
    libnatten kernels are still useful for NAF; the compile-time decorator is not
    required for Pixal3D generation. Keep the patch scoped to the import so the
    rest of PyTorch keeps its normal behavior.
    """

    if os.name != "nt":
        return importlib.import_module(module_name)

    try:
        import torch
    except Exception:
        return importlib.import_module(module_name)

    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return importlib.import_module(module_name)

    def identity_compile(function: Any = None, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if function is None:
            return lambda inner: inner
        return function

    torch.compile = identity_compile
    try:
        return importlib.import_module(module_name)
    finally:
        torch.compile = original_compile


def _install_natten_fallback() -> None:
    """Provide a slow but correct top-level natten fallback when absent.

    Upstream NAF first tries `natten.functional` and falls back to `from natten
    import na2d` on any import error. We intentionally only provide the top-level
    module so the legacy import path still fails and upstream selects the recent
    `na2d(...)` call shape.
    """

    if "natten" in sys.modules:
        return

    try:
        _import_module_with_torch_compile_disabled("natten")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "natten":
            return

    import torch
    import torch.nn.functional as torch_functional

    def na2d(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: Any,
        dilation: Any,
        stride: int = 1,
        backend: str | None = None,
    ) -> torch.Tensor:
        del backend
        if stride != 1:
            raise NotImplementedError("natten fallback only supports stride=1")
        if q.ndim != 5 or k.ndim != 5 or v.ndim != 5:
            raise ValueError("natten fallback expects q/k/v shaped as [b, h, w, n, d]")
        if q.shape != k.shape:
            raise ValueError("natten fallback expects q and k to share the same shape")
        if v.shape[:4] != q.shape[:4]:
            raise ValueError("natten fallback expects v to share q/k [b, h, w, n] dimensions")

        def normalize_pair(value: Any, name: str) -> tuple[int, int]:
            if isinstance(value, int):
                pair = (value, value)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                pair = (int(value[0]), int(value[1]))
            else:
                raise ValueError(f"natten fallback expected {name} to be an int or length-2 tuple")

            if pair[0] <= 0 or pair[1] <= 0:
                raise ValueError(f"natten fallback expected positive {name}")
            return pair

        kernel_h, kernel_w = normalize_pair(kernel_size, "kernel_size")
        dilation_h, dilation_w = normalize_pair(dilation, "dilation")
        if kernel_h % 2 == 0 or kernel_w % 2 == 0:
            raise ValueError("natten fallback requires odd kernel_size values")

        b, h, w, n, d_qk = q.shape
        d_v = v.shape[-1]
        bn = b * n
        neighborhood_size = kernel_h * kernel_w
        padding = ((kernel_h // 2) * dilation_h, (kernel_w // 2) * dilation_w)

        # Process output in two-dimensional tiles, slicing q/k/v in place.
        # Row-only chunks still have a full-width minimum allocation: for the
        # 1024 texture stage even one padded row can cost 146 MiB before unfold.
        # Tiling columns as well removes that minimum while preserving exactly
        # the same neighbourhoods and border mask.
        pad_h, pad_w = padding
        # The broadcasted dot products temporarily coexist with their unfolded
        # K/V neighbourhoods. Count both sides so the target describes peak
        # live storage rather than only the final unfolded tensors.
        bytes_per_position = bn * neighborhood_size * (2 * d_qk + 2 * d_v + 2) * q.element_size()
        tile_h, tile_w = _natten_tile_shape(h, w, bytes_per_position)
        output_shape = (b, h, w, n, d_v)
        output_bytes = _natten_output_size_bytes(output_shape, v.element_size())
        spill_to_cpu = _should_spill_natten_output(v.device.type, output_shape, v.element_size())
        if spill_to_cpu:
            # NAF immediately projects this dense map down to sparse conditioning.
            # Keeping its full-resolution float32 output on CUDA costs 4 GiB at
            # the 1024 texture stage. Spill only the fallback's output storage to
            # system RAM; computation, dtype, weights, resolution, and sampling
            # remain unchanged. The projection compatibility patch below moves
            # only the much smaller projected conditioning back to CUDA.
            print(
                f"[Pixal3D] NATTEN fallback spilling {output_bytes / 2**30:.2f} GiB output to CPU",
                file=sys.stderr,
                flush=True,
            )
            out = torch.empty(output_shape, dtype=v.dtype, device="cpu")
        else:
            out = v.new_empty(output_shape)
        scale = 1.0 / math.sqrt(d_qk)
        unfold_args = {"kernel_size": (kernel_h, kernel_w), "dilation": (dilation_h, dilation_w), "stride": 1}

        def to_heads(x: torch.Tensor, d: int, rows: int, columns: int) -> torch.Tensor:
            return x.permute(0, 3, 4, 1, 2).reshape(bn, d, rows, columns)

        for row_start in range(0, h, tile_h):
            row_end = min(h, row_start + tile_h)
            rows = row_end - row_start
            src_row_start = max(0, row_start - pad_h)
            src_row_end = min(h, row_end + pad_h)
            src_rows = src_row_end - src_row_start

            for column_start in range(0, w, tile_w):
                column_end = min(w, column_start + tile_w)
                columns = column_end - column_start
                src_column_start = max(0, column_start - pad_w)
                src_column_end = min(w, column_end + pad_w)
                src_columns = src_column_end - src_column_start
                pad = (
                    src_column_start - (column_start - pad_w),
                    (column_end + pad_w) - src_column_end,
                    src_row_start - (row_start - pad_h),
                    (row_end + pad_h) - src_row_end,
                )
                positions = rows * columns

                q_dense = to_heads(
                    q[:, row_start:row_end, column_start:column_end],
                    d_qk,
                    rows,
                    columns,
                ).reshape(bn, d_qk, positions).transpose(1, 2)
                k_chunk = torch_functional.pad(
                    to_heads(
                        k[:, src_row_start:src_row_end, src_column_start:src_column_end],
                        d_qk,
                        src_rows,
                        src_columns,
                    ),
                    pad,
                )
                k_neighbors = torch_functional.unfold(k_chunk, **unfold_args).reshape(
                    bn, d_qk, neighborhood_size, positions
                ).permute(0, 3, 2, 1)
                del k_chunk
                ones_chunk = torch_functional.pad(
                    torch.ones((1, 1, src_rows, src_columns), device=q.device, dtype=q.dtype),
                    pad,
                )
                valid_neighbors = torch_functional.unfold(ones_chunk, **unfold_args).reshape(
                    1, neighborhood_size, positions
                ).transpose(1, 2) > 0
                del ones_chunk

                scores = (q_dense.unsqueeze(2) * k_neighbors).sum(dim=-1) * scale
                del k_neighbors, q_dense
                scores.masked_fill_(~valid_neighbors, torch.finfo(scores.dtype).min)
                del valid_neighbors
                attention = torch.softmax(scores, dim=-1)
                del scores
                v_chunk = torch_functional.pad(
                    to_heads(
                        v[:, src_row_start:src_row_end, src_column_start:src_column_end],
                        d_v,
                        src_rows,
                        src_columns,
                    ),
                    pad,
                )
                v_neighbors = torch_functional.unfold(v_chunk, **unfold_args).reshape(
                    bn, d_v, neighborhood_size, positions
                ).permute(0, 3, 2, 1)
                del v_chunk
                chunk = (attention.unsqueeze(-1) * v_neighbors).sum(dim=2)
                del v_neighbors, attention
                chunk = chunk.reshape(b, n, rows, columns, d_v).permute(0, 2, 3, 1, 4)
                if spill_to_cpu:
                    chunk = chunk.to(device="cpu")
                out[:, row_start:row_end, column_start:column_end] = chunk
                del chunk

        return out

    natten_module = types.ModuleType("natten")
    natten_module.__loader__ = None
    natten_module.__package__ = "natten"
    natten_module.__spec__ = importlib.machinery.ModuleSpec("natten", loader=None, is_package=False)
    natten_module.HAS_LIBNATTEN = False
    natten_module.na2d = na2d
    sys.modules["natten"] = natten_module


def _natten_output_size_bytes(shape: tuple[int, ...], element_size: int) -> int:
    return math.prod(shape) * int(element_size)


def _natten_tile_shape(height: int, width: int, bytes_per_position: int) -> tuple[int, int]:
    positions = max(1, _NATTEN_CHUNK_TARGET_BYTES // max(1, bytes_per_position))
    tile_width = max(1, min(width, positions))
    tile_height = max(1, min(height, positions // tile_width))
    return tile_height, tile_width


def _should_spill_natten_output(
    device_type: str,
    shape: tuple[int, ...],
    element_size: int,
) -> bool:
    return (
        device_type == "cuda"
        and _natten_output_size_bytes(shape, element_size)
        >= _NATTEN_CPU_SPILL_THRESHOLD_BYTES
    )


def _release_cuda_memory() -> None:
    """Release dead Python objects and cached CUDA blocks at stage boundaries."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _strip_returned_latents(result: Any) -> Any:
    """Drop decoded latents that Modly's GLB export path never consumes."""
    if not isinstance(result, tuple) or len(result) != 2:
        return result
    meshes, latent_result = result
    if not isinstance(latent_result, tuple) or len(latent_result) != 3:
        return result
    shape_slat, tex_slat, resolution = latent_result
    del shape_slat, tex_slat, result, latent_result
    _release_cuda_memory()
    return meshes, (None, None, resolution)


def _install_cpu_spill_projection_patch() -> None:
    """Project CPU-spilled NAF features without recreating the dense CUDA map."""
    try:
        import torch
        from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj
    except ImportError:
        return

    def move(value: Any, device: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(move(item, device) for item in value)
        if isinstance(value, list):
            return [move(item, device) for item in value]
        if isinstance(value, dict):
            return {key: move(item, device) for key, item in value.items()}
        return value

    # Multi-view builds expose ProjGridMV in the same module. Its call signature
    # differs from ProjGrid, so preserve the installed implementation verbatim
    # and only move its tensor arguments/results across the spill boundary.
    for class_name in ("ProjGrid", "ProjGridMV"):
        projection_class = getattr(image_conditioned_proj, class_name, None)
        if projection_class is None:
            continue
        original = projection_class.forward
        if getattr(original, "_modly_cpu_spill", False):
            continue

        @wraps(original)
        def forward(
            self: Any,
            features_map: Any,
            *args: Any,
            __original: Callable[..., Any] = original,
            **kwargs: Any,
        ) -> Any:
            if not isinstance(features_map, torch.Tensor) or features_map.device.type != "cpu":
                return __original(self, features_map, *args, **kwargs)
            target_device = self.grid_points.device
            if target_device.type == "cpu":
                return __original(self, features_map, *args, **kwargs)

            self.cpu()
            try:
                projected = __original(
                    self,
                    features_map,
                    *move(args, "cpu"),
                    **move(kwargs, "cpu"),
                )
            finally:
                self.to(target_device)
            _release_cuda_memory()
            return move(projected, target_device)

        forward._modly_cpu_spill = True
        projection_class.forward = forward


def _install_pipeline_memory_patches() -> None:
    """Apply quality-neutral lifetime management to the installed pipeline."""
    try:
        import torch
        from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    except ImportError:
        return

    _install_cpu_spill_projection_patch()
    pipeline_classes: list[tuple[Any, str]] = [(Pixal3DImageTo3DPipeline, "run")]
    try:
        from pixal3d.pipelines import Pixal3DMVImageTo3DPipeline

        pipeline_classes.append((Pixal3DMVImageTo3DPipeline, "run_mv"))
    except ImportError:
        pass

    for pipeline_class, run_name in pipeline_classes:
        # Each method below finishes a model stage and, in low-VRAM mode, has
        # just moved that stage's model back to CPU. Emptying the allocator here
        # prevents dead stage blocks from overlapping the next model's live allocations.
        for method_name in (
            "preprocess_image",
            "get_proj_cond_ss",
            "get_proj_cond_shape",
            "sample_sparse_structure",
            "sample_shape_slat",
            "sample_tex_slat",
            "decode_shape_slat",
            "decode_tex_slat",
        ):
            original_method = getattr(pipeline_class, method_name, None)
            if not callable(original_method) or getattr(original_method, "_modly_memory_boundary", False):
                continue

            @wraps(original_method)
            def boundary(
                self: Any,
                *args: Any,
                __original: Callable[..., Any] = original_method,
                __method_name: str = method_name,
                **kwargs: Any,
            ) -> Any:
                try:
                    return __original(self, *args, **kwargs)
                finally:
                    # Background removal is complete after preprocessing. Force
                    # it off CUDA even in standard mode before MoGe is loaded.
                    if __method_name == "preprocess_image":
                        rembg_model = getattr(self, "rembg_model", None)
                        if rembg_model is not None:
                            rembg_model.cpu()
                    _release_cuda_memory()

            boundary._modly_memory_boundary = True
            setattr(pipeline_class, method_name, boundary)

        original_run = getattr(pipeline_class, run_name, None)
        if not callable(original_run) or getattr(original_run, "_modly_memory_patch", False):
            continue

        @wraps(original_run)
        def run(
            self: Any,
            *args: Any,
            __original: Callable[..., Any] = original_run,
            **kwargs: Any,
        ) -> Any:
            try:
                with torch.inference_mode():
                    result = __original(self, *args, **kwargs)
            finally:
                # Standard mode otherwise keeps every model resident while CuMesh
                # builds the GLB. Low-VRAM mode is already on CPU; repeating the
                # offload is harmless and keeps both modes consistent.
                try:
                    self.cpu()
                    for attr in (
                        "image_cond_model_ss",
                        "image_cond_model_shape_512",
                        "image_cond_model_shape_1024",
                        "image_cond_model_tex_1024",
                    ):
                        model = getattr(self, attr, None)
                        if model is not None:
                            model.cpu()
                finally:
                    _release_cuda_memory()
            if kwargs.get("return_latent", False):
                result = _strip_returned_latents(result)
            return result

        run._modly_memory_patch = True
        setattr(pipeline_class, run_name, run)


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc) or type(exc).__name__ == "OutOfMemoryError"


def _retry_after_releasing_cuda_cache(function: Callable[..., Any], label: str, *args: Any, **kwargs: Any) -> Any:
    """Release PyTorch's cached GPU blocks, run, and retry once after an OOM."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    try:
        return function(*args, **kwargs)
    except RuntimeError as exc:
        if not _is_cuda_oom(exc):
            raise
        print(f"[Pixal3D] {label} ran out of GPU memory; retrying after releasing cache (local patch)",
              file=sys.stderr, flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return function(*args, **kwargs)


def _guard_mesh_fill_holes() -> None:
    """Retry, then skip, Mesh.fill_holes when CuMesh runs out of GPU memory.

    fill_holes runs on the raw decoded mesh (tens of millions of faces for some
    inputs) before export simplification, and only closes small holes. On a
    16 GB card it can OOM after all models are already offloaded.
    """
    import gc

    try:
        import torch
        from pixal3d.representations.mesh.base import Mesh
    except ImportError:
        # Contract tests and custom inference shims do not ship the real mesh stack.
        return

    original = Mesh.fill_holes
    if getattr(original, "_modly_oom_guard", False):
        return

    def is_oom(exc: BaseException) -> bool:
        return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc)

    def fill_holes(self, *args, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        for attempt in (1, 2):
            try:
                return original(self, *args, **kwargs)
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                del exc
                gc.collect()
                torch.cuda.empty_cache()
                if attempt == 2:
                    print(
                        f"[Pixal3D] fill_holes skipped: out of GPU memory on {len(self.faces):,} faces (local patch)",
                        file=sys.stderr,
                        flush=True,
                    )
        return None

    fill_holes._modly_oom_guard = True
    Mesh.fill_holes = fill_holes


def _silence_flex_gemm_autotuners() -> None:
    """Keep FlexGEMM autotuning/cache enabled but prevent upstream verbose prints.

    Pixal3D upstream's inference module sets FLEX_GEMM_AUTOTUNER_VERBOSE=1 at
    import time. That is useful in a terminal, but Modly's subprocess bridge
    treats stdout as a structured protocol channel. Existing autotuner instances
    capture the verbose flag during construction, so resetting the environment
    alone is not enough after importing inference.py.
    """

    os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = "0"
    try:
        import flex_gemm
        from flex_gemm.utils.autotuner import PersistentCacheAutoTuner
    except Exception:
        return

    def silence_module(module_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            return
        for value in vars(module).values():
            if isinstance(value, PersistentCacheAutoTuner):
                value.verbose = False

    try:
        for package in ("flex_gemm.kernels.triton.spconv", "flex_gemm.kernels.triton.grid_sample"):
            root = importlib.import_module(package)
            package_path = getattr(root, "__path__", None)
            if package_path is None:
                continue
            import pkgutil

            for module in pkgutil.iter_modules(package_path, f"{package}."):
                silence_module(module.name)
        flex_gemm.utils.load_autotune_cache()
    except Exception:
        return


def _failure(code: str, message: str, **extra: Any) -> dict:
    return {"status": "failed", "code": code, "message": message, "generation_allowed": False, **extra}


_RUNTIME_ENV_KEYS = (
    "TEMP",
    "TMP",
    "TMPDIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)
_WINDOWS_DRIVE_PATH_RE = re.compile(r"""['\"](?P<path>[A-Za-z]:(?:[\\/][^'\"]*)?)['\"]""")
_WINDOWS_DRIVE_PATH_FALLBACK_RE = re.compile(r"(?P<path>[A-Za-z]:[\\/]\S*)")


def _selected_runtime_env() -> dict:
    return {key: os.environ[key] for key in _RUNTIME_ENV_KEYS if key in os.environ}


def _runtime_context(job: dict, image_path: Path, output_dir: Path) -> dict:
    return {
        "cwd": os.getcwd(),
        "env": _selected_runtime_env(),
        "input_image": str(image_path),
        "output_dir": str(output_dir),
        "model_source": str(job.get("model_source") or PIXAL3D_MODEL_SOURCE),
    }


def _looks_like_windows_missing_path(exc: Exception) -> bool:
    message = str(exc).lower()
    if "winerror 3" in message:
        return True
    if not isinstance(exc, OSError):
        return False
    return any(
        marker in message
        for marker in (
            "cannot find the path",
            "cannot find path",
            "system cannot find the path specified",
            "impossibile trovare il percorso",
            "non riesce a trovare il percorso",
        )
    )


def _extract_windows_missing_path(exc: Exception) -> str | None:
    if not _looks_like_windows_missing_path(exc):
        return None
    message = str(exc)
    match = _WINDOWS_DRIVE_PATH_RE.search(message) or _WINDOWS_DRIVE_PATH_FALLBACK_RE.search(message)
    if match is None:
        return None
    return match.group("path").rstrip(".,;)")


def _runtime_failure(exc: Exception, checkpoint: str, job: dict, image_path: Path, output_dir: Path) -> dict:
    result = _failure(
        "runtime_failed",
        str(exc),
        checkpoint=checkpoint,
        exception_type=type(exc).__name__,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        runtime_context=_runtime_context(job, image_path, output_dir),
    )
    failed_path = _extract_windows_missing_path(exc)
    if failed_path is not None:
        result["path_hint"] = {
            "code": "missing_windows_path",
            "message": "A configured runtime/cache/model/input/output path appears to reference a missing Windows drive or root.",
            "path": failed_path,
        }
    return result


def _parse_low_vram(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "low_vram", "low-vram", "low vram"}:
            return True
        if normalized in {"false", "0", "no", "off", "standard"}:
            return False
        return default
    return default


def _parse_texture_size(value: Any, default: int = DEFAULT_TEXTURE_SIZE) -> int:
    safe_default = default if default in SUPPORTED_TEXTURE_SIZES else DEFAULT_TEXTURE_SIZE
    if value is None:
        return safe_default
    if isinstance(value, bool):
        return safe_default
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return safe_default
    else:
        return safe_default
    return parsed if parsed in SUPPORTED_TEXTURE_SIZES else safe_default


def _parse_manual_fov(value: Any, default: float = DEFAULT_MANUAL_FOV) -> float:
    safe_default = DEFAULT_MANUAL_FOV
    try:
        parsed_default = float(default)
    except (TypeError, ValueError):
        parsed_default = DEFAULT_MANUAL_FOV
    if math.isfinite(parsed_default):
        for allowed in SUPPORTED_MANUAL_FOVS:
            if math.isclose(parsed_default, allowed, rel_tol=0.0, abs_tol=1e-9):
                safe_default = allowed
                break

    if value is None:
        return safe_default
    if isinstance(value, bool):
        return safe_default
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return safe_default
    else:
        return safe_default

    if not math.isfinite(parsed):
        return safe_default
    for allowed in SUPPORTED_MANUAL_FOVS:
        if math.isclose(parsed, allowed, rel_tol=0.0, abs_tol=1e-9):
            return allowed
    return safe_default


@contextmanager
def _scoped_texture_size_env(texture_size: Any):
    parsed_texture_size = _parse_texture_size(texture_size)
    # The environment is process-global and the exporter override below is
    # process-global module state. Serialize both with one re-entrant lock so
    # concurrent jobs cannot observe each other's requested texture size.
    with _TEXTURE_EXPORT_LOCK:
        previous = os.environ.get(PIXAL3D_TEXTURE_SIZE_ENV)
        os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = str(parsed_texture_size)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(PIXAL3D_TEXTURE_SIZE_ENV, None)
            else:
                os.environ[PIXAL3D_TEXTURE_SIZE_ENV] = previous


@contextmanager
def _scoped_to_glb_texture_size(texture_size: Any, inference_module: Any | None = None):
    """Force the tracked selector into the upstream Linux and Windows exporter."""

    parsed_texture_size = _parse_texture_size(texture_size)
    o_voxel_module = getattr(inference_module, "o_voxel", None)
    postprocess = getattr(o_voxel_module, "postprocess", None)
    if postprocess is None:
        # Contract tests and custom inference shims do not load o_voxel. The
        # real upstream inference module imports it before this context starts.
        yield
        return
    if not callable(getattr(postprocess, "to_glb", None)):
        raise RuntimeError("Loaded o_voxel runtime does not expose postprocess.to_glb")

    with _TEXTURE_EXPORT_LOCK:
        original = postprocess.to_glb

        def selected_to_glb(*args, **kwargs):
            kwargs["texture_size"] = parsed_texture_size
            # CuMesh allocates with raw cudaMalloc, so blocks PyTorch cached
            # during decoding are invisible to it until they are released.
            return _retry_after_releasing_cuda_cache(original, "to_glb", *args, **kwargs)

        postprocess.to_glb = selected_to_glb
        try:
            yield
        finally:
            postprocess.to_glb = original


@contextmanager
def _scoped_single_view_texture_export(texture_size: Any, inference_module: Any | None = None):
    """Serialize base export state against the complete multiview runtime."""
    from pixal3d_extension.multiview import _MV_RUN_LOCK

    # Both requests patch the same process-global environment/module state.
    # Acquire the outer MV lock first so a base request cannot install its
    # wrapper while multiview inference is still exporting. The nested scopes
    # use RLocks because single-view NAF setup re-enters _MV_RUN_LOCK below.
    with _MV_RUN_LOCK, _scoped_texture_size_env(texture_size), _scoped_to_glb_texture_size(
        texture_size, inference_module
    ):
        yield


def _resolve_glb(output_dir: Path, pipeline_result: Any) -> Path | None:
    if isinstance(pipeline_result, dict) and pipeline_result.get("glb_path"):
        candidate = Path(pipeline_result["glb_path"])
        return candidate if candidate.is_file() and is_contained_path(output_dir, candidate) else None
    matches = sorted(output_dir.glob("*.glb"))
    return matches[0] if matches else None


def _check_cancel(cancel_event: Any | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Pixal3D generation cancelled")


def _accepts_keyword(callable_object: Any, keyword: str) -> bool:
    """Return whether a runtime callable can receive an optional host keyword."""
    try:
        parameters = inspect.signature(callable_object).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters)


def _publish_staged_glb(staged: Path, output_dir: Path) -> Path:
    """Atomically promote one regular, non-empty GLB from private staging."""
    if staged.is_symlink():
        raise RuntimeError("Pixal3D runtime produced a symlinked GLB output")
    staged = staged.resolve(strict=True)
    if not staged.is_file() or staged.stat().st_size <= 0:
        raise RuntimeError("Pixal3D runtime produced an invalid GLB output")
    final = output_dir / staged.name
    if final.exists() or final.is_symlink():
        raise RuntimeError(f"Pixal3D output already exists: {final.name}")
    os.replace(staged, final)
    return final


def _resolve_workspace_output_dir(workspace_root: str | Path, output_dir: str) -> Path | dict:
    try:
        candidate = require_contained_path(workspace_root, output_dir, allow_absolute=True)
    except ValueError:
        return _failure("unsafe_path", "output_dir must be contained in the workspace root")

    base = Path(workspace_root)
    allowed_roots = (base / "outputs", base / "workspace")
    if not any(is_contained_path(allowed_root, candidate) for allowed_root in allowed_roots):
        return _failure("unsafe_path", "output_dir must be contained in the workspace outputs or workspace root")
    return candidate


def _resolve_job_paths(job: dict) -> tuple[Path, Path] | dict:
    workspace_root = job.get("workspace_root")
    if not workspace_root:
        image_path = Path(job.get("input_image", ""))
        output_dir = Path(job.get("output_dir", ""))
        if not str(output_dir):
            return _failure("output_missing", "output_dir is required")
        return image_path, output_dir

    try:
        image_path = require_contained_path(workspace_root, job.get("input_image", ""), allow_absolute=True)
    except ValueError:
        return _failure("unsafe_path", "input_image must be contained in the workspace root")

    output_dir = _resolve_workspace_output_dir(workspace_root, job.get("output_dir", ""))
    if isinstance(output_dir, dict):
        return output_dir

    return image_path, output_dir


def _job_auxiliary_mode(job: dict) -> str:
    params = job.get("params") or {}
    if job.get("offline") is True or params.get("offline") is True:
        return "offline"
    return str(job.get("auxiliary_mode") or params.get("auxiliary_mode") or "default")


def _job_network_available(job: dict, auxiliary_mode: str) -> bool:
    params = job.get("params") or {}
    if "network_available" in job:
        return bool(job["network_available"])
    if "network_available" in params:
        return bool(params["network_available"])
    return auxiliary_mode not in {"offline"}


def _job_auxiliary_bootstrap_downloader(job: dict) -> Any:
    params = job.get("params") or {}
    return job.get("auxiliary_bootstrap_downloader") or params.get("auxiliary_bootstrap_downloader")


def _job_modly_home(job: dict) -> str | Path | None:
    return derive_modly_home(
        model_dir=job.get("model_source"),
        workspace_dir=job.get("workspace_root") or job.get("output_dir"),
    ) or job.get("workspace_root")


def _with_auxiliary_bootstrap_metadata(auxiliary_source: dict, bootstrap_result: dict | None) -> dict:
    if bootstrap_result is None:
        return auxiliary_source
    warnings = list(auxiliary_source.get("warnings", []))
    if bootstrap_result.get("status") != "ready":
        warnings.append(
            {
                "code": "auxiliary_bootstrap_failed_remote_fallback_preserved",
                "message": "explicit first-run auxiliary bootstrap failed; preserving remote/HF-cache fallback",
            }
        )
    enriched = {**auxiliary_source, "auxiliary_bootstrap": bootstrap_result}
    if warnings:
        enriched["warnings"] = warnings
    return enriched


def _patch_pipeline_for_runtime(
    workspace_root: str | Path,
    *,
    auxiliary_mode: str,
    network_available: bool,
    auxiliary_source: dict,
    bootstrap_result: dict | None,
) -> tuple[dict | None, dict | None]:
    patch_result = patch_pipeline(
        workspace_root,
        auxiliary_mode=auxiliary_mode,
        network_available=network_available,
    )
    patched_auxiliary_source = _with_auxiliary_bootstrap_metadata(
        patch_result.get("auxiliary_source", auxiliary_source),
        bootstrap_result,
    )
    if patch_result.get("status") != "patched":
        return (
            _failure(
                patch_result.get("code", "pipeline_substitution_required"),
                patch_result.get("message", "Pixal3D pipeline substitutions are required before generation"),
                auxiliary_mode=auxiliary_mode,
                pipeline_patch=patch_result,
                auxiliary_source=patched_auxiliary_source,
            ),
            None,
        )
    return None, patched_auxiliary_source


def _preflight_auxiliary_sources(job: dict) -> tuple[dict | None, dict | None]:
    auxiliary_mode = _job_auxiliary_mode(job)
    try:
        normalized_auxiliary_mode = normalize_auxiliary_source_mode(auxiliary_mode)
        local_required = requires_local_auxiliary(normalized_auxiliary_mode)
    except ValueError as exc:
        return _failure("invalid_auxiliary_mode", str(exc), auxiliary_mode=auxiliary_mode), None
    network_available = _job_network_available(job, normalized_auxiliary_mode)
    workspace_root = _job_modly_home(job)

    if not workspace_root:
        if local_required:
            return (
                _failure(
                    "missing_auxiliary_assets",
                    "workspace_root is required to validate local Pixal3D auxiliary assets",
                    auxiliary_mode=normalized_auxiliary_mode,
                    missing=[],
                ),
                None,
            )
        return None, None

    try:
        auxiliary_source = resolve_auxiliary_sources(
            workspace_root,
            mode=normalized_auxiliary_mode,
            network_available=network_available,
        )
    except ValueError as exc:
        return _failure("invalid_auxiliary_mode", str(exc), auxiliary_mode=normalized_auxiliary_mode), None

    if auxiliary_source["status"] == "blocked":
        return (
            _failure(
                "missing_auxiliary_assets",
                "local Pixal3D auxiliary assets are missing for the requested mode",
                auxiliary_mode=normalized_auxiliary_mode,
                missing=auxiliary_source.get("missing", []),
                auxiliary_source=auxiliary_source,
            ),
            None,
        )

    bootstrap_result = None
    if normalized_auxiliary_mode == "default" and network_available and auxiliary_source.get("status") == "fallback":
        bootstrap_result = bootstrap_auxiliary_assets(
            workspace_root,
            downloader=_job_auxiliary_bootstrap_downloader(job),
        )
        if bootstrap_result.get("status") != "ready":
            return (
                _failure(
                    "naf_bootstrap_failed",
                    "NAF checkpoint download failed; retry generation or run setup.py --bootstrap-auxiliary-assets manually",
                    auxiliary_bootstrap=bootstrap_result,
                ),
                None,
            )
        auxiliary_source = resolve_auxiliary_sources(
            workspace_root,
            mode=normalized_auxiliary_mode,
            network_available=network_available,
        )
        patch_error, patched_auxiliary_source = _patch_pipeline_for_runtime(
            workspace_root,
            auxiliary_mode=normalized_auxiliary_mode,
            network_available=network_available,
            auxiliary_source=auxiliary_source,
            bootstrap_result=bootstrap_result,
        )
        if patch_error is not None:
            return patch_error, None
        return None, patched_auxiliary_source

    patch_result = validate_pipeline_patch(
        workspace_root,
        auxiliary_mode=normalized_auxiliary_mode,
        network_available=network_available,
    )
    if patch_result["status"] != "ready":
        return (
            _failure(
                patch_result.get("code", "pipeline_substitution_required"),
                patch_result.get("message", "Pixal3D pipeline substitutions are required before generation"),
                auxiliary_mode=normalized_auxiliary_mode,
                pipeline_patch=patch_result,
                auxiliary_source=auxiliary_source,
            ),
            None,
        )

    unresolved_runtime_dependencies = auxiliary_source.get("unlocalized_runtime_dependencies", [])
    if normalized_auxiliary_mode in {"offline", "strict"} and unresolved_runtime_dependencies and not bool(job.get("allow_remote_runtime_dependencies", False)):
        return (
            _failure(
                "offline_runtime_dependencies_unresolved",
                "Some Pixal3D runtime dependencies still rely on cache or network fallback behavior.",
                auxiliary_mode=normalized_auxiliary_mode,
                auxiliary_source=auxiliary_source,
                runtime_dependencies=auxiliary_source.get("unlocalized_runtime_dependencies", []),
            ),
            None,
        )

    return None, auxiliary_source


def _preflight_runtime(job: dict) -> tuple[dict | None, dict | None]:
    readiness = job.get("readiness")
    if readiness is None:
        return _failure("readiness_required", "Pixal3D readiness must pass before generation"), None
    if readiness.get("generation_allowed") is not True:
        result = _failure(
            readiness.get("code", "readiness_required"),
            readiness.get("message", "Pixal3D readiness must pass before generation"),
        )
        if "dependency_diagnostics" in readiness:
            result["dependency_diagnostics"] = readiness["dependency_diagnostics"]
        return result, None

    auxiliary_error, auxiliary_source = _preflight_auxiliary_sources(job)
    if auxiliary_error is not None:
        return auxiliary_error, None

    runtime_lane = job.get("runtime_lane")
    if runtime_lane and runtime_lane not in SUPPORTED_RUNTIME_LANES | EXPERIMENTAL_BASE_RUNTIME_LANES:
        return _failure("unsupported_lane", f"runtime lane {runtime_lane!r} is not eligible for base generation"), auxiliary_source

    asset_readiness = job.get("asset_readiness") or {}
    if asset_readiness.get("code") == "missing_assets" or job.get("assets_ready") is False:
        result = _failure("missing_assets", "required Pixal3D assets are missing")
        result["missing"] = asset_readiness.get("missing", [])
        return result, auxiliary_source

    return None, auxiliary_source


def _patch_inference_dino_source(inference_module: Any, auxiliary_source: dict) -> None:
    dino_source = auxiliary_source.get("sources", {}).get("dino", {})
    if dino_source.get("kind") != "local":
        return

    local_dino_path = str(dino_source["value"])
    configs = getattr(inference_module, "IMAGE_COND_CONFIGS", None)
    if isinstance(configs, dict):
        iterable = configs.values()
    elif isinstance(configs, (list, tuple)):
        iterable = configs
    else:
        return

    for config in iterable:
        if isinstance(config, dict) and "model_name" in config:
            config["model_name"] = local_dino_path
        elif hasattr(config, "model_name"):
            setattr(config, "model_name", local_dino_path)


def _patch_inference_moge_loader(inference_module: Any, auxiliary_source: dict) -> None:
    moge_source = auxiliary_source.get("sources", {}).get("moge", {})
    if moge_source.get("kind") != "local":
        return

    local_moge_path = str(moge_source["value"])
    load_moge_model = getattr(inference_module, "load_moge_model", None)
    if not callable(load_moge_model):
        return

    setattr(inference_module, "MOGE_MODEL_NAME", local_moge_path)

    def load_local_moge_model(*args: Any, **kwargs: Any) -> Any:
        args_list = list(args)
        if len(args_list) >= 2:
            args_list[1] = local_moge_path
            kwargs.pop("model_name", None)
        else:
            kwargs["model_name"] = local_moge_path
        return load_moge_model(*args_list, **kwargs)

    load_local_moge_model.__name__ = getattr(load_moge_model, "__name__", "load_moge_model")
    load_local_moge_model.__doc__ = getattr(load_moge_model, "__doc__", None)
    setattr(inference_module, "load_moge_model", load_local_moge_model)


def _patch_inference_auxiliary_sources(inference_module: Any, auxiliary_source: dict | None) -> None:
    if not auxiliary_source:
        return
    _patch_inference_dino_source(inference_module, auxiliary_source)
    _patch_inference_moge_loader(inference_module, auxiliary_source)


@contextmanager
def _scoped_single_view_naf_extractors(inference_module: Any, auxiliary_source: dict | None):
    """Route upstream extractor NAF calls to the verified local checkpoint only."""
    from pixal3d_extension.multiview import _MV_RUN_LOCK, _local_naf_extractors

    naf_source = (auxiliary_source or {}).get("sources", {}).get("naf", {})
    if naf_source.get("kind") != "local":
        raise RuntimeError("single-view inference requires a local NAF checkpoint")
    if not callable(getattr(inference_module, "build_image_cond_model", None)):
        raise RuntimeError("single-view inference has no image-condition extractor factory")
    with _MV_RUN_LOCK, _local_naf_extractors(
        inference_module, Path(naf_source["value"]).resolve(), verify_checkpoint=True
    ):
        yield


def run_job(
    job: dict,
    *,
    pipeline_factory: Callable[[str], Any] | None = None,
    cancel_event: Any | None = None,
) -> dict:
    _enable_crash_diagnostics()
    _diagnostic_checkpoint("run_job:start")
    resolved_paths = _resolve_job_paths(job)
    if isinstance(resolved_paths, dict):
        return resolved_paths
    image_path, output_dir = resolved_paths

    if not image_path.is_file():
        return _failure("invalid_image", "input_image must reference an existing local image file")

    preflight_error, auxiliary_source = _preflight_runtime(job)
    if preflight_error is not None:
        return preflight_error

    checkpoint = "runtime_block"
    try:
        _check_cancel(cancel_event)
        checkpoint = "preflight_ok"
        _diagnostic_checkpoint("run_job:preflight:ok")
        params = job.get("params") or {}
        parsed_low_vram = _parse_low_vram(params.get("low_vram"), default=True)
        parsed_texture_size = _parse_texture_size(params.get("texture_size"), default=DEFAULT_TEXTURE_SIZE)
        parsed_manual_fov = _parse_manual_fov(params.get("manual_fov"), default=DEFAULT_MANUAL_FOV)
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pixal3d-run-", dir=output_dir.parent) as temporary:
            staging = Path(temporary)
            if pipeline_factory is not None:
                checkpoint = "pipeline_factory_create"
                _diagnostic_checkpoint("pipeline_factory:create:start")
                pipeline = pipeline_factory(job.get("model_source") or PIXAL3D_MODEL_SOURCE)
                _diagnostic_checkpoint("pipeline_factory:create:done")
                checkpoint = "pipeline_factory_call"
                _diagnostic_checkpoint("pipeline_factory:call:start")
                pipeline_kwargs = {
                    "image_path": str(image_path),
                    "output_dir": str(staging),
                    "seed": params.get("seed"),
                    "resolution": params.get("resolution", 1024),
                    "low_vram": parsed_low_vram,
                    "texture_size": parsed_texture_size,
                    "manual_fov": parsed_manual_fov,
                }
                if _accepts_keyword(pipeline, "cancel_event"):
                    pipeline_kwargs["cancel_event"] = cancel_event
                result = pipeline(**pipeline_kwargs)
                _diagnostic_checkpoint("pipeline_factory:call:done")
            else:
                checkpoint = "runtime_compat"
                _diagnostic_checkpoint("runtime_compat:start")
                _prepare_runtime_compat()
                _diagnostic_checkpoint("runtime_compat:done")
                checkpoint = "windows_aliases"
                _diagnostic_checkpoint("windows_aliases:start")
                _install_windows_native_module_aliases()
                _diagnostic_checkpoint("windows_aliases:done")
                checkpoint = "natten"
                _diagnostic_checkpoint("natten:start")
                _install_natten_fallback()
                _diagnostic_checkpoint("natten:done")
                checkpoint = "import_inference"
                _diagnostic_checkpoint("import_inference:start")
                inference_module = importlib.import_module("inference")
                _patch_inference_auxiliary_sources(inference_module, auxiliary_source)
                _install_pipeline_memory_patches()
                run_inference = inference_module.run_inference
                _diagnostic_checkpoint("import_inference:done")

                checkpoint = "flex_gemm_silence"
                _silence_flex_gemm_autotuners()
                _diagnostic_checkpoint("flex_gemm_silence:done")
                _guard_mesh_fill_holes()

                checkpoint = "prepare_inference_args"
                seed = int(params.get("seed", -1))
                if seed == -1:
                    seed = random.randint(0, 2**32 - 1)
                staged_glb = staging / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_pixal3d.glb"
                checkpoint = "run_inference"
                _diagnostic_checkpoint("run_inference:start")
                inference_kwargs = {
                    "image_path": str(image_path),
                    "output_path": str(staged_glb),
                    "seed": seed,
                    "model_path": job.get("model_source") or PIXAL3D_MODEL_SOURCE,
                    "manual_fov": parsed_manual_fov,
                    "low_vram": parsed_low_vram,
                    "resolution": int(params.get("resolution", 1024)),
                }
                if _accepts_keyword(run_inference, "cancel_event"):
                    inference_kwargs["cancel_event"] = cancel_event
                with _scoped_single_view_texture_export(
                    parsed_texture_size, inference_module
                ), _scoped_single_view_naf_extractors(inference_module, auxiliary_source):
                    run_inference(**inference_kwargs)
                _diagnostic_checkpoint("run_inference:done")
                result = {"glb_path": str(staged_glb), "pbr": {}}

            _check_cancel(cancel_event)
            staged_glb = _resolve_glb(staging, result)
            if staged_glb is None:
                raise RuntimeError("Pixal3D runtime did not produce a GLB output")
            glb_path = _publish_staged_glb(staged_glb, output_dir)
    except Exception as exc:  # pragma: no cover - contract retained for real runtime failures.
        _diagnostic_checkpoint(f"runtime_exception:{type(exc).__name__}")
        return _runtime_failure(exc, checkpoint, job, image_path, output_dir)

    pbr = result.get("pbr", {}) if isinstance(result, dict) else {}
    return {
        "status": "completed",
        "model_source": PIXAL3D_MODEL_SOURCE,
        "auxiliary_source": auxiliary_source,
        "output": {"glb_path": str(glb_path), "pbr": pbr},
        "params": {
            "seed": params.get("seed"),
            "resolution": params.get("resolution", 1024),
            "low_vram": parsed_low_vram,
            "texture_size": parsed_texture_size,
            "manual_fov": parsed_manual_fov,
        },
    }
