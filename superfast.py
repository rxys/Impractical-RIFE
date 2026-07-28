"""HF cache for Smoothie's TensorRT dependencies and compiled cores."""

import hashlib, importlib.metadata, importlib.util, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

__all__ = ["load", "save"]

_root, _cores = Path("/content/smoothie-trt"), Path("/content/vs_rife_benchmark/engines-flow-v2")
_ready = _root / "inference-ready"
_token = _remote = _tag = None
_info, _original_cores, _cache_hit = {}, set(), False


def _identity():
    import torch
    info = dict(
        version=3,
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
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


def _install_builder():
    _deactivate()
    py = [sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir"]
    try:
        codec_ready = importlib.metadata.version("torchcodec") == "0.13.0+cu126"
    except importlib.metadata.PackageNotFoundError:
        codec_ready = False
    if not codec_ready:
        subprocess.run(py + ["--no-deps",
            "https://download.pytorch.org/whl/cu126/"
            "torchcodec-0.13.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"], check=True)
    if all(importlib.util.find_spec(name) for name in ("torch_tensorrt", "vapoursynth", "vsrife")):
        return
    subprocess.run(py + ["--no-deps", "torch-tensorrt==2.11.0",
                   "--extra-index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run(py + ["tensorrt-cu12>=10.15.1,<10.16", "vapoursynth==77",
                   "vsrife==5.7.0", "vidgear", "dllist",
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
    _remote = f"hf://buckets/{bucket_name}/smoothie/trt-bundles/v1/{_tag}.zip"
    archive = Path(f"/content/smoothie-trt-{_tag}.zip")
    result = _hf(_remote, archive, check=False)
    _cache_hit = result.returncode == 0
    if not _cache_hit:
        print("No compatible cache; installing the TensorRT builder once.")
        _install_builder()
        return
    shutil.rmtree(_root, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(_root)
    archive.unlink()
    if json.loads((_root / "compatibility.json").read_text()) != _info:
        raise RuntimeError("Incompatible TensorRT cache")
    _cores.mkdir(parents=True, exist_ok=True)
    for source in (_root / "engines").glob("*"):
        shutil.copy2(source, _cores / source.name)
    _original_cores = {p.name for p in _cores.glob("*")}
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
    if _cache_hit and set(current) == _original_cores:
        return
    if not (_root / "runtime").is_dir():
        _copy_runtime()
    engines = _root / "engines"
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
