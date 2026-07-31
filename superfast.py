"""Portable HF cache for Smoothie's TensorRT runtime and compiled engines."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "load",
    "start_load",
    "wait_for_load",
    "run_inference",
    "save",
]

ENGINE_DIR_ENV = "SMOOTHIE_ENGINE_DIR"

_CACHE_VERSION = 5
_TORCH_TRT_VERSION = "2.11.0"
_TENSORRT_VERSION = "10.14.1.48.post1"
_VAPOURSYNTH_VERSION = "77"
_VSRIFE_VERSION = "5.7.0"
_TORCHCODEC_VERSION = "0.13.0+cu126"

_token: str | None = None
_remote: str | None = None
_tag: str | None = None
_root: Path | None = None
_cores: Path | None = None
_info: dict[str, Any] = {}
_original_cores: dict[str, int] = {}
_cache_hit = False

_state = threading.Condition(threading.RLock())
_load_started = False
_load_finished = False
_load_error: BaseException | None = None

_inference_started = False
_inference_engine_pending = False
_inference_error: BaseException | None = None

_engine_operation_active = False
_engine_operation_completed = False
_engine_operation_source: str | None = None
_engine_completed_source: str | None = None
_engine_error: BaseException | None = None


def _workspace() -> Path:
    """Return the process workspace selected when the cache was first used."""
    global _root
    if _root is None:
        _root = Path.cwd().resolve()
    return _root


def _engine_dir(*, reset_to_workspace: bool = False) -> Path:
    """Resolve and publish the shared engine directory."""
    global _cores
    if reset_to_workspace or _cores is None:
        if reset_to_workspace or not os.environ.get(ENGINE_DIR_ENV):
            path = _workspace() / "engines"
            os.environ[ENGINE_DIR_ENV] = str(path)
        else:
            path = Path(os.environ[ENGINE_DIR_ENV]).expanduser().resolve()
        _cores = path
    _cores.mkdir(parents=True, exist_ok=True)
    return _cores


def _identity() -> tuple[dict[str, Any], str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    info = dict(
        version=_CACHE_VERSION,
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        torch_tensorrt=_TORCH_TRT_VERSION,
        tensorrt=_TENSORRT_VERSION,
        vapoursynth=_VAPOURSYNTH_VERSION,
        vsrife=_VSRIFE_VERSION,
    )
    tag = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
    return info, tag


def _hf(*args: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"HF_XET_HIGH_PERFORMANCE": "1"}
    if _token:
        env["HF_TOKEN"] = _token
    return subprocess.run(
        ["hf", "buckets", "cp", *map(str, args)],
        env=env,
        capture_output=not check,
        text=True,
        check=check,
    )


def _remove_path_from_env(name: str, fragment: Path) -> None:
    target = str(fragment)
    os.environ[name] = os.pathsep.join(
        value
        for value in os.getenv(name, "").split(os.pathsep)
        if value and target not in value
    )


def _deactivate() -> None:
    runtime = _workspace() / "runtime"
    sys.path[:] = [entry for entry in sys.path if str(runtime) not in str(entry)]
    _remove_path_from_env("PYTHONPATH", runtime)
    _remove_path_from_env("LD_LIBRARY_PATH", runtime)
    importlib.invalidate_caches()


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_builder() -> None:
    """Install compiler-time dependencies when a matching engine is absent."""
    _deactivate()
    py = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade"]
    if _installed_version("torchcodec") != _TORCHCODEC_VERSION:
        subprocess.run(
            py
            + [
                "--no-deps",
                f"torchcodec=={_TORCHCODEC_VERSION}",
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
            ],
            check=True,
        )
    expected = {
        "torch-tensorrt": _TORCH_TRT_VERSION,
        "tensorrt-cu12": _TENSORRT_VERSION,
        "vapoursynth": _VAPOURSYNTH_VERSION,
        "vsrife": _VSRIFE_VERSION,
    }
    if all(_installed_version(name) == version for name, version in expected.items()):
        return
    subprocess.run(
        py
        + [
            "--no-deps",
            f"torch-tensorrt=={_TORCH_TRT_VERSION}",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        check=True,
    )
    subprocess.run(
        py
        + [
            f"tensorrt-cu12=={_TENSORRT_VERSION}",
            f"vapoursynth=={_VAPOURSYNTH_VERSION}",
            f"vsrife=={_VSRIFE_VERSION}",
            "vidgear",
            "dllist",
            "--extra-index-url",
            "https://pypi.nvidia.com",
        ],
        check=True,
    )
    subprocess.run(["vapoursynth", "config"], check=True)
    importlib.invalidate_caches()


def _activate() -> None:
    root = _workspace()
    package_candidates = sorted(
        {
            path.resolve()
            for pattern in (
                "runtime/**/python*/site-packages",
                "runtime/**/python*/dist-packages",
            )
            for path in root.glob(pattern)
            if path.is_dir()
        }
    )
    if not package_candidates:
        raise RuntimeError("Cached Python runtime directory was not found")
    libraries = {
        str(path.parent.resolve())
        for pattern in (
            "runtime/**/site-packages/nvidia/**/lib*.so*",
            "runtime/**/site-packages/tensorrt_libs/lib*.so*",
            "runtime/**/site-packages/tensorrt_cu12_libs/lib*.so*",
            "runtime/**/dist-packages/nvidia/**/lib*.so*",
            "runtime/**/dist-packages/tensorrt_libs/lib*.so*",
            "runtime/**/dist-packages/tensorrt_cu12_libs/lib*.so*",
        )
        for path in root.glob(pattern)
    }

    package_texts = [str(path) for path in package_candidates]
    for package_text in reversed(package_texts):
        if package_text not in sys.path:
            sys.path.insert(0, package_text)
    current_pythonpath = [
        entry for entry in os.getenv("PYTHONPATH", "").split(os.pathsep) if entry
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            *package_texts,
            *[entry for entry in current_pythonpath if entry not in package_texts],
        ]
    )
    current_ld = [
        entry for entry in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*sorted(libraries), *[entry for entry in current_ld if entry not in libraries]]
    )
    importlib.invalidate_caches()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in TensorRT cache: {member.filename}")
        bundle.extractall(destination)


def _snapshot_cores() -> dict[str, int]:
    return {
        path.name: path.stat().st_size
        for path in _engine_dir().glob("*")
        if path.is_file() and path.stat().st_size
    }


def _clear_managed_cache() -> None:
    root = _workspace()
    _deactivate()
    shutil.rmtree(root / "runtime", ignore_errors=True)
    shutil.rmtree(root / "engines", ignore_errors=True)
    (root / "compatibility.json").unlink(missing_ok=True)
    global _cores
    _cores = root / "engines"
    os.environ[ENGINE_DIR_ENV] = str(_cores)
    _cores.mkdir(parents=True, exist_ok=True)


def _claim_load() -> bool:
    global _load_started, _load_finished, _load_error
    with _state:
        if _load_started:
            return False
        _load_started = True
        _load_finished = False
        _load_error = None
        _state.notify_all()
        return True


def _finish_load(error: BaseException | None) -> None:
    global _load_finished, _load_error
    with _state:
        _load_error = error
        _load_finished = True
        _state.notify_all()


def wait_for_load() -> None:
    """Wait for a currently running cache load; return immediately if none started."""
    with _state:
        if not _load_started:
            return
        _state.wait_for(lambda: _load_finished)
        error = _load_error
    if error is not None:
        raise error


# Private alias used by inference_video without exposing synchronization details.
def _wait_for_load() -> None:
    wait_for_load()


def _notify_inference_started() -> None:
    global _inference_started, _inference_engine_pending, _inference_error
    with _state:
        _inference_started = True
        _inference_engine_pending = True
        _inference_error = None
        _state.notify_all()


def _notify_inference_failed(error: BaseException) -> None:
    global _inference_engine_pending, _inference_error
    with _state:
        _inference_engine_pending = False
        _inference_error = error
        _state.notify_all()


def _begin_engine_operation(source: str) -> None:
    """Serialize the one engine load/build phase shared by load and inference."""
    global _engine_operation_active, _engine_operation_source
    global _engine_error, _inference_engine_pending
    with _state:
        _state.wait_for(lambda: not _engine_operation_active)
        if source == "inference":
            _inference_engine_pending = False
        _engine_operation_active = True
        _engine_operation_source = source
        _engine_error = None
        _state.notify_all()


def _finish_engine_operation(error: BaseException | None = None) -> None:
    global _engine_operation_active, _engine_operation_completed
    global _engine_operation_source, _engine_completed_source, _engine_error
    with _state:
        completed_source = _engine_operation_source
        _engine_operation_active = False
        _engine_operation_source = None
        _engine_error = error
        if error is None:
            _engine_operation_completed = True
            _engine_completed_source = completed_source
        _state.notify_all()


def _prebuild(width: int, height: int, pad_multiple: int) -> None:
    _begin_engine_operation("load")
    error: BaseException | None = None
    try:
        import inference_video

        inference_video.ensure_vsrife_engines(
            width,
            height,
            pad_multiple=pad_multiple,
        )
    except BaseException as caught:
        error = caught
        raise
    finally:
        _finish_engine_operation(error)


def _load_impl(
    token: str | None,
    bucket_name: str | None,
    width: int | None,
    height: int | None,
    pad_multiple: int,
) -> None:
    global _token, _info, _tag, _remote, _original_cores, _cache_hit

    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if width is not None and (width <= 0 or height is None or height <= 0):
        raise ValueError("width and height must be greater than zero")
    if pad_multiple <= 0:
        raise ValueError("pad_multiple must be greater than zero")

    _token, _remote, _cache_hit = token, None, False
    root = _workspace()
    cores = _engine_dir(reset_to_workspace=True)

    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        if not shutil.which("apt-get"):
            raise RuntimeError("ffmpeg and ffprobe are required")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    if bucket_name and not shutil.which("hf"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            check=True,
        )

    _info, _tag = _identity()
    if not bucket_name:
        print(
            "\033[1;91mWARNING: No HF bucket configured. Normal dependency "
            "installation and TensorRT compilation will be slow.\033[0m"
        )
        _install_builder()
        _original_cores = _snapshot_cores()
        if width is not None and height is not None:
            _prebuild(width, height, pad_multiple)
        return

    _remote = f"hf://buckets/{bucket_name}/smoothie/trt-bundles/v2/{_tag}.zip"
    archive = root / f"smoothie-trt-{_tag}.zip"
    result = _hf(_remote, archive, check=False)
    _cache_hit = result.returncode == 0

    if not _cache_hit:
        print("No compatible cache; installing the TensorRT builder once.")
        archive.unlink(missing_ok=True)
        _install_builder()
        _original_cores = _snapshot_cores()
        if width is not None and height is not None:
            _prebuild(width, height, pad_multiple)
        return

    try:
        _clear_managed_cache()
        _safe_extract(archive, root)
        archive.unlink(missing_ok=True)
        if json.loads((root / "compatibility.json").read_text()) != _info:
            raise RuntimeError("Incompatible TensorRT cache")
        _activate()
    except Exception as error:
        print(f"Ignoring unusable TensorRT cache: {error}")
        archive.unlink(missing_ok=True)
        _clear_managed_cache()
        _cache_hit = False
        _install_builder()

    cores.mkdir(parents=True, exist_ok=True)
    _original_cores = _snapshot_cores()
    if _cache_hit:
        print(f"TensorRT cache restored ({len(_original_cores)} core files).")
    if width is not None and height is not None:
        _prebuild(width, height, pad_multiple)


def load(
    token: str | None = None,
    bucket_name: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    pad_multiple: int = 64,
) -> None:
    """Restore dependencies and optionally prepare the engine for video dimensions."""
    if not _claim_load():
        wait_for_load()
        return

    error: BaseException | None = None
    try:
        _load_impl(token, bucket_name, width, height, pad_multiple)
    except BaseException as caught:
        error = caught
        raise
    finally:
        _finish_load(error)


def start_load(
    token: str | None = None,
    bucket_name: str | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
    pad_multiple: int = 64,
) -> concurrent.futures.Future[None]:
    """Start ``load`` in a daemon thread and return a Future for synchronization."""
    future: concurrent.futures.Future[None] = concurrent.futures.Future()

    if not _claim_load():
        def wait_existing() -> None:
            try:
                wait_for_load()
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(None)

        threading.Thread(target=wait_existing, name="smoothie-load-wait", daemon=True).start()
        return future

    # Bind cwd and publish the engine path before the worker can race with the
    # caller's preparation work or a later inference call.
    try:
        _workspace()
        _engine_dir(reset_to_workspace=True)
    except BaseException as error:
        _finish_load(error)
        future.set_exception(error)
        return future

    def worker() -> None:
        error: BaseException | None = None
        try:
            _load_impl(token, bucket_name, width, height, pad_multiple)
        except BaseException as caught:
            error = caught
            future.set_exception(caught)
        else:
            future.set_result(None)
        finally:
            _finish_load(error)

    threading.Thread(target=worker, name="smoothie-load", daemon=True).start()
    return future


def run_inference(argv: Sequence[str] | None = None) -> Any:
    """Wait for cache loading, then run inference in the foreground."""
    wait_for_load()
    import inference_video

    return inference_video.main(argv)


def _copy_runtime(runtime: Path | None = None) -> None:
    runtime = runtime or (_workspace() / "runtime")
    for name in (
        "torchcodec",
        "torch-tensorrt",
        "tensorrt",
        "tensorrt-cu12",
        "tensorrt-cu12-bindings",
        "tensorrt-cu12-libs",
        "vidgear",
        "colorlog",
    ):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in dist.files or ():
            source = Path(dist.locate_file(item))
            skip = "builder_resource" in source.name.lower() or source.suffix in {
                ".h",
                ".hpp",
                ".a",
            }
            if not source.is_file() or skip:
                continue
            resolved = source.resolve()
            try:
                relative = resolved.relative_to(_workspace() / "runtime")
            except ValueError:
                try:
                    relative = resolved.relative_to("/")
                except ValueError:
                    continue
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _wait_until_saveable() -> None:
    """Wait for a completed prebuild or for inference's engine phase to settle."""
    with _state:
        _state.wait_for(lambda: _load_started or _inference_started)
        if _load_started:
            _state.wait_for(lambda: _load_finished)
            if _load_error is not None:
                raise _load_error

        while True:
            if _engine_error is not None:
                raise _engine_error
            if _inference_error is not None and not _engine_operation_completed:
                raise _inference_error
            if (
                _engine_operation_completed
                and _engine_completed_source == "load"
                and not _engine_operation_active
            ):
                return
            if (
                _engine_operation_completed
                and _engine_completed_source == "inference"
                and not _engine_operation_active
                and not _inference_engine_pending
            ):
                return
            if (
                _inference_started
                and not _inference_engine_pending
                and not _engine_operation_active
            ):
                return
            _state.wait()


def _has_complete_engine_pair(cores: Path) -> bool:
    flow_engines = [
        path
        for path in cores.glob("*.ts")
        if path.is_file() and not path.name.endswith(".encode") and path.stat().st_size
    ]
    return any(
        (Path(str(flow_path) + ".encode")).is_file()
        and (Path(str(flow_path) + ".encode")).stat().st_size
        for flow_path in flow_engines
    )


def save() -> None:
    """Upload changed engines once the sole engine load/build phase is stable."""
    global _original_cores

    _wait_until_saveable()
    if not _remote or not _tag:
        return

    cores = _engine_dir()
    current = _snapshot_cores()
    if not _has_complete_engine_pair(cores):
        raise RuntimeError("Inference started without a complete TensorRT engine pair")
    if _cache_hit and current == _original_cores:
        return

    root = _workspace()
    # Copy before replacing the managed runtime so active cached packages can
    # still be used as sources if no builder installation was necessary.
    runtime = root / "runtime"
    runtime_next = root / ".smoothie-runtime-next"
    shutil.rmtree(runtime_next, ignore_errors=True)
    _copy_runtime(runtime_next)
    shutil.rmtree(runtime, ignore_errors=True)
    runtime_next.rename(runtime)
    compatibility = root / "compatibility.json"
    compatibility.write_text(json.dumps(_info, sort_keys=True))

    archive = root / f"smoothie-trt-{_tag}.zip"
    try:
        with zipfile.ZipFile(
            archive,
            "w",
            zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as bundle:
            runtime = root / "runtime"
            if runtime.exists():
                for path in runtime.rglob("*"):
                    if path.is_file():
                        bundle.write(path, path.relative_to(root))
            for path in cores.glob("*"):
                if path.is_file():
                    bundle.write(path, Path("engines") / path.name)
            bundle.write(compatibility, compatibility.name)
        size_mib = archive.stat().st_size / 2**20
        _hf(archive, _remote)
        _original_cores = current
        print(f"TensorRT cache saved ({size_mib:.1f} MiB).")
    finally:
        archive.unlink(missing_ok=True)
