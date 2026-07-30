"""HF cache for Smoothie's TensorRT dependencies and compiled cores."""

import hashlib, importlib.metadata, importlib.util, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

__all__ = ["load", "save"]

_root, _cores = Path("/content/smoothie-trt"), Path("/content/vs_rife_benchmark/engines-flow-v2")
_ready = _root / "inference-ready"
_token = _remote = _tag = None
_info, _original_cores, _cache_hit = {}, {}, False

_CACHE_VERSION = 4
_TORCH_TRT_VERSION = "2.11.0"
_TENSORRT_VERSION = "10.14.1.48.post1"
_VAPOURSYNTH_VERSION = "77"
_VSRIFE_VERSION = "5.7.0"
_TORCHCODEC_VERSION = "0.13.0+cu126"


def _identity():
    import torch
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


def _hf(*args, check=True):
    env = os.environ | dict(HF_XET_HIGH_PERFORMANCE="1")
    if _token:
        env["HF_TOKEN"] = _token
    return subprocess.run(["hf", "buckets", "cp", *map(str, args)], env=env,
                          capture_output=not check, text=True, check=check)


def _deactivate():
    sys.path[:] = [p for p in sys.path if str(_root / "runtime") not in p]
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        os.environ[name] = os.pathsep.join(
            p for p in os.getenv(name, "").split(os.pathsep) if str(_root) not in p
        )


def _installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_builder():
    _deactivate()
    py = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade"]
    if _installed_version("torchcodec") != _TORCHCODEC_VERSION:
        subprocess.run(py + ["--no-deps",
            "https://download.pytorch.org/whl/cu126/"
            "torchcodec-0.13.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"], check=True)
    expected = {
        "torch-tensorrt": _TORCH_TRT_VERSION,
        "tensorrt-cu12": _TENSORRT_VERSION,
        "vapoursynth": _VAPOURSYNTH_VERSION,
        "vsrife": _VSRIFE_VERSION,
    }
    if all(_installed_version(name) == version for name, version in expected.items()):
        return
    subprocess.run(py + ["--no-deps", f"torch-tensorrt=={_TORCH_TRT_VERSION}",
                   "--extra-index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run(py + [f"tensorrt-cu12=={_TENSORRT_VERSION}",
                   f"vapoursynth=={_VAPOURSYNTH_VERSION}",
                   f"vsrife=={_VSRIFE_VERSION}", "vidgear", "dllist",
                   "--extra-index-url", "https://pypi.nvidia.com"], check=True)
    subprocess.run(["vapoursynth", "config"], check=True)


def _activate():
    packages = next(_root.glob("runtime/usr/local/lib/python*/dist-packages"))
    libraries = {str(p.parent) for pattern in (
        "runtime/usr/local/lib/python*/dist-packages/nvidia/**/lib*.so*",
        "runtime/usr/local/lib/python*/dist-packages/tensorrt_libs/lib*.so*",
    ) for p in _root.glob(pattern)}
    os.environ["PYTHONPATH"] = os.pathsep.join([str(packages), os.getenv("PYTHONPATH", "")])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        sorted(libraries) + ([os.environ["LD_LIBRARY_PATH"]] if os.getenv("LD_LIBRARY_PATH") else [])
    )


def load(token=None, bucket_name=None):
    global _token, _info, _tag, _remote, _original_cores, _cache_hit
    _token, _remote, _cache_hit = token, None, False
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)
    if bucket_name and not shutil.which("hf"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            check=True,
        )
    if not bucket_name:
        print(
            "\033[1;91mWARNING: No HF bucket configured. Normal dependency "
            "installation and TensorRT compilation will be slow.\033[0m"
        )
        _install_builder()
        return
    _root.mkdir(parents=True, exist_ok=True)
    _ready.unlink(missing_ok=True)
    os.environ["SMOOTHIE_CACHE_READY"] = str(_ready)
    _info, _tag = _identity()
    _remote = f"hf://buckets/{bucket_name}/smoothie/trt-bundles/v2/{_tag}.zip"
    archive = Path(f"/content/smoothie-trt-{_tag}.zip")
    result = _hf(_remote, archive, check=False)
    _cache_hit = result.returncode == 0
    if not _cache_hit:
        print("No compatible cache; installing the TensorRT builder once.")
        _install_builder()
        return
    try:
        shutil.rmtree(_root, ignore_errors=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(_root)
        archive.unlink()
        if json.loads((_root / "compatibility.json").read_text()) != _info:
            raise RuntimeError("Incompatible TensorRT cache")
    except Exception as error:
        print(f"Ignoring unusable TensorRT cache: {error}")
        archive.unlink(missing_ok=True)
        shutil.rmtree(_root, ignore_errors=True)
        _root.mkdir(parents=True, exist_ok=True)
        _cache_hit = False
        _install_builder()
        return
    shutil.rmtree(_cores, ignore_errors=True)
    _cores.mkdir(parents=True, exist_ok=True)
    for source in (_root / "engines").glob("*"):
        shutil.copy2(source, _cores / source.name)
    _original_cores = {p.name: p.stat().st_size for p in _cores.glob("*") if p.stat().st_size}
    _activate()
    print(f"TensorRT cache restored ({len(_original_cores)} core files).")


def _copy_runtime():
    runtime = _root / "runtime"
    for name in ("torchcodec", "torch-tensorrt", "tensorrt", "tensorrt-cu12",
                 "tensorrt-cu12-bindings", "tensorrt-cu12-libs", "vidgear", "colorlog"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for item in dist.files or ():
            source = Path(dist.locate_file(item))
            skip = ("builder_resource" in source.name.lower() or
                    source.suffix in {".h", ".hpp", ".a"})
            if not source.is_file() or skip:
                continue
            try:
                target = runtime / source.resolve().relative_to("/")
            except ValueError:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def save():
    if not _remote:
        return
    inference_started = False
    for _ in range(36000):
        if _ready.exists():
            break
        running = subprocess.run(
            ["pgrep", "-f", "[i]nference_video.py"],
            stdout=subprocess.DEVNULL,
        ).returncode == 0
        inference_started |= running
        if inference_started and not running:
            return
        time.sleep(0.2)
    else:
        return
    current = {p.name: p.stat().st_size for p in _cores.glob("*") if p.stat().st_size}
    if not any(name.endswith(".encode") for name in current):
        raise RuntimeError("Inference started without a complete TensorRT core")
    if _cache_hit and current == _original_cores:
        return
    # Engines may have just been rebuilt after restoring an older cache. Always
    # package the runtime currently installed beside those engines; retaining
    # the restored runtime can create a mixed, non-deserializable archive.
    shutil.rmtree(_root / "runtime", ignore_errors=True)
    _copy_runtime()
    engines = _root / "engines"
    shutil.rmtree(engines, ignore_errors=True)
    engines.mkdir(parents=True, exist_ok=True)
    for source in _cores.glob("*"):
        shutil.copy2(source, engines / source.name)
    (_root / "compatibility.json").write_text(json.dumps(_info, sort_keys=True))
    archive = Path(f"/content/smoothie-trt-{_tag}.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in _root.rglob("*"):
            if path.is_file() and path != _ready:
                z.write(path, path.relative_to(_root))
    _hf(archive, _remote)
    print(f"TensorRT cache saved ({archive.stat().st_size / 2**20:.1f} MiB).")
