import re
import os
import ctypes
import importlib.util
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import subprocess
from queue import Queue
from vidgear.gears import VideoGear
import math
import time as walltime
from fractions import Fraction
from pathlib import Path

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None, required=True)
parser.add_argument('--scene_video', dest='scene_video', type=str, default=None, help='Low-res video specifically for ffmpeg scene detection')
parser.add_argument('--scene_detector', dest='scene_detector', type=str, default='hash', choices=['hash', 'ffmpeg', 'none'], help='Scene detector to use. "hash" runs on the already decoded frames.')
parser.add_argument('--scene_hash_threshold', dest='scene_hash_threshold', type=float, default=0.410, help='Normalized Hamming distance threshold for hash scene detection')
parser.add_argument('--scene_hash_size', dest='scene_hash_size', type=int, default=16, help='Low-frequency DCT square size for hash scene detection')
parser.add_argument('--scene_hash_lowpass', dest='scene_hash_lowpass', type=int, default=2, help='DCT lowpass factor for hash scene detection')
parser.add_argument('--scene_min_len', dest='scene_min_len', type=float, default=0.5, help='Minimum seconds between scene cuts')
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--fps', dest='fps', type=float, default=None, required=True)
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--drop_input', dest='drop_input', type=int, default=1, help='Only keep every Nth input frame (1 = keep all, 2 = drop every other, etc.)')
parser.add_argument('--fixed_height', type=int, default=None, help='Fixed vertical resolution for downscaling while keeping aspect ratio')
parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug visualization')
parser.add_argument('--av1', dest='use_av1', action='store_true', help='Use GPU AV1 encoding (av1_nvenc) instead of h264_nvenc')
parser.add_argument('--out_chunks', dest='out_chunks', action='store_true', help='Output streamable chunks via segment muxer')
parser.add_argument('--range', dest='frame_range', type=int, nargs=2, metavar=('START', 'END'), help='Process source-frame interval [START, END); boundaries must align to output frames')
parser.add_argument('--gop', type=int, default=None, help='GOP size for ranged chunk output')
parser.add_argument('--dedup', dest='dedup', action='store_true', help='Drop duplicate frames before interpolation to restore smooth motion')
parser.add_argument('--dedup_global_thresh', dest='dedup_global_thresh', type=float, default=0.5, help='Global MAD threshold for dedup on 64x64 frame')
parser.add_argument('--dedup_block_thresh', dest='dedup_block_thresh', type=float, default=2.0, help='Max 8x8 block MAD threshold for dedup on 64x64 frame')
parser.add_argument('--batch_timestamps', dest='batch_timestamps', type=int, default=1, help='Batch this many interpolation timestamps for the same source-frame pair; benchmark before increasing')
parser.add_argument('--pad_multiple', dest='pad_multiple', type=int, default=64, choices=[64, 128], help='Spatial padding alignment')

args = parser.parse_args()

if args.frame_range is not None:
    range_start, range_end = args.frame_range
    if range_start < 0 or range_end <= range_start:
        parser.error("--range requires 0 <= START < END")
    if args.out_chunks and args.gop is None:
        parser.error("--gop is required with --range --out_chunks")
elif args.gop is not None:
    parser.error("--gop is only valid with --range")
if args.gop is not None and args.gop <= 0:
    parser.error("--gop must be greater than zero")
if args.batch_timestamps <= 0:
    parser.error("--batch_timestamps must be greater than zero")
if args.batch_timestamps != 1:
    parser.error("VS-RIFE TensorRT currently requires --batch_timestamps=1")

from model.warplayer import warp

def forward_warp(img, flow):
    """
    Bilinear normalized forward splat.

    img:  [B, C, H, W]
    flow: [B, 2, H, W], source -> destination displacement in pixels
    """
    B, C, H, W = img.shape

    # Use FP32 for coordinates and accumulation, even during FP16 inference.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing="ij",
    )

    tx = xx.unsqueeze(0) + flow[:, 0].float()
    ty = yy.unsqueeze(0) + flow[:, 1].float()

    x0 = torch.floor(tx)
    y0 = torch.floor(ty)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    src = img.float().reshape(B, C, -1)
    accum = torch.zeros(
        B, C, H * W, device=img.device, dtype=torch.float32
    )
    weight_sum = torch.zeros(
        B, 1, H * W, device=img.device, dtype=torch.float32
    )

    def splat(x, y, weight):
        valid = (
            (x >= 0) & (x < W) &
            (y >= 0) & (y < H)
        )

        # Clamp only to produce safe indices. Invalid contributions get zero weight.
        xi = x.clamp(0, W - 1).long()
        yi = y.clamp(0, H - 1).long()

        idx = (yi * W + xi).reshape(B, 1, -1)
        wgt = (
            weight * valid.to(weight.dtype)
        ).reshape(B, 1, -1)

        accum.scatter_add_(
            2,
            idx.expand(-1, C, -1),
            src * wgt,
        )
        weight_sum.scatter_add_(2, idx, wgt)

    splat(x0, y0, (x1 - tx) * (y1 - ty))
    splat(x1, y0, (tx - x0) * (y1 - ty))
    splat(x0, y1, (x1 - tx) * (ty - y0))
    splat(x1, y1, (tx - x0) * (ty - y0))

    accum = accum.reshape(B, C, H, W)
    weight_sum = weight_sum.reshape(B, 1, H, W)

    result = accum / weight_sum.clamp_min(1e-6)
    result = result.to(img.dtype)

    # Approximate inverse warp for newly exposed holes.
    backward_x = 2.0 * (xx.unsqueeze(0) - flow[:, 0].float()) / max(W - 1, 1) - 1.0
    backward_y = 2.0 * (yy.unsqueeze(0) - flow[:, 1].float()) / max(H - 1, 1) - 1.0
    backward_grid = torch.stack((backward_x, backward_y), dim=-1)
    fallback = F.grid_sample(
        img.float(),
        backward_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).to(img.dtype)

    return torch.where(weight_sum > 1e-6, result, fallback)


def forward_monkey(
    self,
    x,
    timestep=0.5,
    scale_list=[16, 8, 4, 2, 1],
    training=False,
    fastmode=True,
    ensemble=False,
    cached_f0=None,
    cached_f1=None,
):
    if training == False:
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

    # Extrapolation for timestep > 1
    if not training and isinstance(timestep, float) and timestep > 1.0:
        # Midpoint gives two useful, reasonably balanced intermediate flows.
        flow_list, _, merged = self.forward(
            x,
            timestep=0.5,
            scale_list=scale_list,
            training=False,
            fastmode=True,
            ensemble=False,
        )

        flow_mid = flow_list[-1]

        # These are midpoint -> img0 and midpoint -> img1 sampling vectors.
        mid_to_0 = flow_mid[:, :2]
        mid_to_1 = flow_mid[:, 2:4]

        # Under constant motion:
        # mid_to_0 = -0.5 * velocity
        # mid_to_1 = +0.5 * velocity
        velocity_mid = mid_to_1 - mid_to_0

        # Relocate the velocity field from midpoint coordinates onto img1.
        velocity_at_img1 = forward_warp(velocity_mid, mid_to_1)

        d = timestep - 1.0
        extrapolated_frame = forward_warp(img1, d * velocity_at_img1)

        merged[-1] = extrapolated_frame
        return None, None, merged

    if not torch.is_tensor(timestep):
        timestep = (x[:, :1].clone() * 0 + 1) * timestep
    elif timestep.shape[2:] != img0.shape[2:]:
        timestep = timestep.expand(-1, -1, img0.shape[2], img0.shape[3])
    f0 = self.encode(img0[:, :3]) if cached_f0 is None else cached_f0
    f1 = self.encode(img1[:, :3]) if cached_f1 is None else cached_f1
    flow_list = []
    merged = []
    mask_list = []
    warped_img0 = img0
    warped_img1 = img1
    flow = None
    mask = None
    loss_cons = 0
    block = [self.block0, self.block1, self.block2, self.block3, self.block4]
    for i in range(5):
        if flow is None:
            flow, mask, feat = block[i](torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1), None, scale=scale_list[i])
            if ensemble:
                print("warning: ensemble is not supported since RIFEv4.21")
        else:
            wf0 = warp(f0, flow[:, :2])
            wf1 = warp(f1, flow[:, 2:4])
            fd, m0, feat = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], wf0, wf1, timestep, mask, feat), 1), flow, scale=scale_list[i])
            if ensemble:
                print("warning: ensemble is not supported since RIFEv4.21")
            else:
                mask = m0
            flow = flow + fd
        mask_list.append(mask)
        flow_list.append(flow)
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        merged.append((warped_img0, warped_img1))
    mask = torch.sigmoid(mask)
    merged[4] = (warped_img0 * mask + warped_img1 * (1 - mask))
    if not fastmode:
        print('contextnet is removed')
        '''
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged[4] = torch.clamp(merged[4] + res, 0, 1)
        '''
    return flow_list, mask_list[4], merged


if not torch.cuda.is_available():
    parser.error("CUDA is required")
device = torch.device("cuda")
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

flow_engine = None
encode_engine = None
ten_flow_div = None
warp_grid = None
timestamp_tensors = {}
cached_f0 = None
cached_f1 = None
cached_pair_key = None
previous_pair_img1 = None
previous_pair_f1 = None

def trt_warp(img, flow, flow_div, warp_grid):
    dtype = img.dtype
    flow = flow.float() / flow_div.view(1, 2, 1, 1)
    grid = (warp_grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        img.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).to(dtype)

def trt_forward_with_flow(self, img0, img1, timestep, flow_div, warp_grid, f0, f1):
    img0 = img0.clamp(0.0, 1.0)
    img1 = img1.clamp(0.0, 1.0)
    warped0, warped1 = img0, img1
    flow = mask = None
    for i, block in enumerate((self.block0, self.block1, self.block2, self.block3, self.block4)):
        if flow is None:
            flow, mask, feat = block(
                torch.cat((img0, img1, f0, f1, timestep), 1),
                None,
                scale=self.scale_list[i],
            )
        else:
            wf0 = trt_warp(f0, flow[:, :2], flow_div, warp_grid)
            wf1 = trt_warp(f1, flow[:, 2:4], flow_div, warp_grid)
            delta, mask, feat = block(
                torch.cat((warped0, warped1, wf0, wf1, timestep, mask, feat), 1),
                flow,
                scale=self.scale_list[i],
            )
            flow = flow + delta
        warped0 = trt_warp(img0, flow[:, :2], flow_div, warp_grid)
        warped1 = trt_warp(img1, flow[:, 2:4], flow_div, warp_grid)
    mask = torch.sigmoid(mask)
    return warped0 * mask + warped1 * (1 - mask), flow

def ensure_vsrife_engines(img0, img1):
    global flow_engine, encode_engine, ten_flow_div, warp_grid
    if flow_engine is not None:
        return

    setup_started = walltime.perf_counter()
    print(f"[trt] loading engines for {img0.shape[-1]}x{img0.shape[-2]}", flush=True)
    import gc

    model_version = "4.26"
    padded_height, padded_width = img0.shape[-2:]
    workspace_bytes = 4 * (1 << 30)
    engine_dir = Path("/content/vs_rife_benchmark/engines-flow-v2")
    engine_dir.mkdir(parents=True, exist_ok=True)

    def find_flow_engines():
        return sorted(
            path
            for path in engine_dir.glob(
                f"flownet_v{model_version}.pkl_"
                f"{padded_width}x{padded_height}_fp16_*_"
                f"workspace-{workspace_bytes}_level-5.ts"
            )
            if not path.name.endswith(".encode")
        )

    engine_paths = find_flow_engines()
    if not engine_paths:
        print(
            f"[trt] cached cores do not include {padded_width}x{padded_height}; "
            "installing the builder",
            flush=True,
        )
        try:
            import superfast
            superfast._install_builder()
        except ImportError:
            pass
        import torch_tensorrt
        # These packages are compiler/build-time dependencies only. Keeping them
        # out of the cached-engine path makes a small runtime artifact possible.
        import vapoursynth as vs
        import vsrife
        from vsrife import rife
        from vsrife.__main__ import download_model
        from vsrife.IFNet_HDv3_v4_26 import IFNet

        IFNet.forward = trt_forward_with_flow

        model_dir = Path(vsrife.__file__).resolve().parent / "models"
        model_path = model_dir / f"flownet_v{model_version}.pkl"
        if not model_path.exists() or model_path.stat().st_size == 0:
            download_model(
                "https://github.com/HolyWu/vs-rife/releases/download/model/"
                f"flownet_v{model_version}.pkl"
            )
        print(
            f"Building VS-RIFE TensorRT engines for "
            f"{padded_width}x{padded_height}; this is a one-time setup cost."
        )
        core = vs.core
        blank = core.std.BlankClip(
            width=padded_width,
            height=padded_height,
            length=2,
            format=vs.RGBH,
            fpsnum=24,
            fpsden=1,
            keep=True,
        )
        built = rife(
            blank,
            device_index=0,
            model=model_version,
            auto_download=False,
            fps_num=60,
            fps_den=1,
            scale=1.0,
            ensemble=False,
            sc=False,
            trt=True,
            trt_static_shape=True,
            trt_workspace_size=workspace_bytes,
            trt_max_aux_streams=None,
            trt_optimization_level=5,
            trt_cache_dir=str(engine_dir),
        )
        del built, blank
        if hasattr(core, "clear_cache"):
            core.clear_cache()
        gc.collect()
        torch.cuda.empty_cache()
        engine_paths = find_flow_engines()
    else:
        # Loading serialized Torch-TensorRT engines only needs the runtime
        # registration library. Avoid importing the compiler-facing Python
        # package and its ONNX parser dependency on cached-engine startups.
        runtime_entries = {
            Path(entry)
            for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if entry and Path(entry).is_dir()
        }
        runtime_spec = importlib.util.find_spec("torch_tensorrt")
        if runtime_spec and runtime_spec.submodule_search_locations:
            runtime_entries.update(Path(entry).parent for entry in runtime_spec.submodule_search_locations)
        runtime_libraries = [
            path
            for entry in runtime_entries
            for path in (
                entry / "torch_tensorrt" / "lib" / "libtorchtrt_runtime.so",
                entry / "torch_tensorrt" / "lib" / "libtorchtrt_plugins.so",
            )
            if path.is_file()
        ]
        if not runtime_libraries:
            raise RuntimeError("Torch-TensorRT runtime libraries were not found")
        native_libraries = []
        for entry in runtime_entries:
            for folder in (
                entry / "tensorrt_libs",
                entry / "tensorrt_cu12_libs",
                entry / "nvidia" / "tensorrt" / "lib",
            ):
                if folder.is_dir():
                    native_libraries.extend(folder.glob("libnvinfer.so*"))
                    native_libraries.extend(folder.glob("libnvinfer_plugin.so*"))
        for library in sorted(set(native_libraries), key=lambda path: "plugin" in path.name):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        for library in runtime_libraries:
            torch.ops.load_library(str(library))

    assert len(engine_paths) == 1, (
        f"Expected one matching VS-RIFE flow engine, found: {engine_paths}"
    )
    flow_path = engine_paths[0]
    encode_path = Path(str(flow_path) + ".encode")
    assert encode_path.is_file(), f"Missing VS-RIFE encode engine: {encode_path}"

    flow_engine = torch.jit.load(str(flow_path)).eval()
    encode_engine = torch.jit.load(str(encode_path)).eval()
    ten_flow_div = torch.tensor(
        [(padded_width - 1.0) / 2.0, (padded_height - 1.0) / 2.0],
        device=device,
        dtype=torch.float32,
    )
    horizontal = torch.linspace(
        -1.0, 1.0, padded_width, device=device, dtype=torch.float32
    ).view(1, 1, 1, padded_width).expand(-1, -1, padded_height, -1)
    vertical = torch.linspace(
        -1.0, 1.0, padded_height, device=device, dtype=torch.float32
    ).view(1, 1, padded_height, 1).expand(-1, -1, -1, padded_width)
    warp_grid = torch.cat([horizontal, vertical], 1)

    warm_f0 = encode_engine(img0)
    warm_f1 = encode_engine(img1)
    warm_timestep = torch.full(
        (1, 1, padded_height, padded_width),
        0.5,
        device=device,
        dtype=img0.dtype,
    )
    for _ in range(4):
        flow_engine(
            img0,
            img1,
            warm_timestep,
            ten_flow_div,
            warp_grid,
            warm_f0,
            warm_f1,
        )
    torch.cuda.synchronize()
    print(
        f"VS-RIFE TensorRT load/build+warmup: "
        f"{walltime.perf_counter() - setup_started:.2f}s"
    )

def get_pair_features(img0, img1):
    global cached_f0, cached_f1, cached_pair_key
    global previous_pair_img1, previous_pair_f1
    ensure_vsrife_engines(img0, img1)
    key = (
        img0.data_ptr(),
        img1.data_ptr(),
        tuple(img0.shape),
        tuple(img1.shape),
    )
    if key != cached_pair_key:
        cached_f0 = (
            previous_pair_f1
            if img0 is previous_pair_img1
            else encode_engine(img0)
        )
        cached_f1 = encode_engine(img1)
        previous_pair_img1 = img1
        previous_pair_f1 = cached_f1
        cached_pair_key = key
    return cached_f0, cached_f1

def run_inference(img0, img1, timestep, inference_scale=1):
    if inference_scale != 1:
        raise ValueError("VS-RIFE TensorRT supports inference_scale=1 only")
    if torch.is_tensor(timestep):
        if timestep.numel() != 1:
            raise ValueError("VS-RIFE TensorRT uses one timestamp per execution")
        timestamp = float(timestep.item())
    else:
        timestamp = float(timestep)

    f0, f1 = get_pair_features(img0, img1)
    timestamp_key = round(timestamp, 8)
    timestep_tensor = timestamp_tensors.get(timestamp_key)
    if timestep_tensor is None:
        timestep_tensor = torch.full(
            (1, 1, img0.shape[-2], img0.shape[-1]),
            timestamp,
            device=img0.device,
            dtype=img0.dtype,
        )
        timestamp_tensors[timestamp_key] = timestep_tensor
    if timestamp > 1.0:
        midpoint = timestamp_tensors.get(0.5)
        if midpoint is None:
            midpoint = torch.full_like(timestep_tensor, 0.5)
            timestamp_tensors[0.5] = midpoint
        _, flow = flow_engine(
            img0, img1, midpoint, ten_flow_div, warp_grid, f0, f1
        )
        velocity_mid = flow[:, 2:4] - flow[:, :2]
        velocity_at_img1 = forward_warp(velocity_mid, flow[:, 2:4])
        result = forward_warp(img1, (timestamp - 1.0) * velocity_at_img1)
    else:
        result, _ = flow_engine(
            img0, img1, timestep_tensor, ten_flow_div, warp_grid, f0, f1
        )
    ready = os.environ.pop("SMOOTHIE_CACHE_READY", None)
    if ready:
        Path(ready).touch()
    return result

videoCapture = cv2.VideoCapture(args.video)
source_fps = videoCapture.get(cv2.CAP_PROP_FPS) / args.drop_input
timestep = source_fps / args.fps
tot_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
videoCapture.release()

range_start_output = range_end_output = expected_range_frames = None
chunk_frames = segment_frames = None
decode_start = 0
if args.frame_range is not None:
    ratio = Fraction(args.fps / source_fps).limit_denominator(1000)
    if not math.isclose(float(ratio), args.fps / source_fps, rel_tol=1e-7, abs_tol=1e-7):
        parser.error("could not resolve an exact source/output frame ratio")
    range_start, range_end = args.frame_range
    if range_start % ratio.denominator or range_end % ratio.denominator:
        parser.error(
            f"--range boundaries must be multiples of {ratio.denominator} source frames "
            f"for the {ratio.numerator}/{ratio.denominator} output ratio"
        )
    range_start_output = range_start * ratio.numerator // ratio.denominator
    range_end_output = range_end * ratio.numerator // ratio.denominator
    expected_range_frames = range_end_output - range_start_output
    decode_start = max(0, range_start - 1)

    if args.out_chunks:
        chunk_unit = math.lcm(args.gop, ratio.numerator)
        chunk_frames = max(chunk_unit, int(args.fps * 10 / chunk_unit + 0.5) * chunk_unit)
        cuts = range(chunk_frames, expected_range_frames, chunk_frames)
        segment_frames = ",".join(map(str, cuts)) or str(chunk_frames)

print(
    f"[decode] opening {args.video} at source frame {decode_start} "
    f"for range {args.frame_range}",
    flush=True,
)
videogen = VideoGear(
    source=args.video,
    backend='ffmpeg',
    **({"CAP_PROP_POS_FRAMES": decode_start} if decode_start else {}),
).start()
first_frame = videogen.read()
print(f"[decode] first frame received: {first_frame is not None}", flush=True)
if first_frame is None:
    raise RuntimeError(f"Decoder returned no frame at source frame {decode_start}")
lastframe = first_frame.copy() if first_frame is not None else None
video_path_wo_ext, ext = os.path.splitext(args.video)
h, w, _ = lastframe.shape
if args.fixed_height is not None:
    orig_h, orig_w = h, w
    new_h = args.fixed_height
    new_w = int(orig_w * (new_h / orig_h))
    h, w = new_h, new_w
    lastframe = cv2.resize(lastframe, (w, h), interpolation=cv2.INTER_AREA)
print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frames, source_fps, args.fps))

class HashSceneDetector:
    def __init__(self, threshold, size, lowpass, min_scene_len_frames):
        self.threshold = threshold
        self.size = size
        self.size_sq = float(size * size)
        self.lowpass = lowpass
        self.min_scene_len_frames = min_scene_len_frames
        self.last_hash = None
        self.last_scene_cut = 0
        self.last_score = None

    @staticmethod
    def hash_frame(frame_img, hash_size, factor):
        gray_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        imsize = hash_size * factor
        resized_img = cv2.resize(gray_img, (imsize, imsize), interpolation=cv2.INTER_AREA)
        max_value = np.max(resized_img)
        if max_value == 0:
            max_value = 1
        resized_img = np.float32(resized_img) / max_value
        dct_complete = cv2.dct(resized_img)
        dct_low_freq = dct_complete[:hash_size, :hash_size]
        med = np.median(np.asarray(dct_low_freq, dtype=np.float32))
        return dct_low_freq > med

    def process_frame(self, frame_img, frame_index):
        curr_hash = self.hash_frame(frame_img, self.size, self.lowpass)
        if self.last_hash is None:
            self.last_hash = curr_hash
            self.last_score = None
            return False

        hash_dist = np.count_nonzero(curr_hash != self.last_hash)
        hash_dist_norm = hash_dist / self.size_sq
        self.last_hash = curr_hash
        self.last_score = hash_dist_norm

        if (
            hash_dist_norm >= self.threshold
            and frame_index - self.last_scene_cut >= self.min_scene_len_frames
        ):
            self.last_scene_cut = frame_index
            return True
        return False

class TorchHashSceneDetector:
    def __init__(self, threshold, size, lowpass, min_scene_len_frames):
        self.threshold = threshold
        self.size = size
        self.size_sq = float(size * size)
        self.lowpass = lowpass
        self.min_scene_len_frames = min_scene_len_frames
        self.last_hash = None
        self.last_hash_cpu = None
        self.last_scene_cut = 0
        self.last_score = None
        self._shape = None
        self._wy = None
        self._wx = None
        self._dct = None

    @staticmethod
    def _area_weights(input_size, output_size, device):
        scale = input_size / output_size
        source = torch.arange(input_size, device=device, dtype=torch.float32)
        starts = torch.arange(output_size, device=device, dtype=torch.float32) * scale
        ends = starts + scale
        overlap = torch.minimum(ends[:, None], source[None, :] + 1.0)
        overlap -= torch.maximum(starts[:, None], source[None, :])
        return overlap.clamp_(0.0, 1.0) / scale

    def _prepare(self, height, width, device):
        shape = (height, width, device)
        if self._shape == shape:
            return
        output_size = self.size * self.lowpass
        self._wy = self._area_weights(height, output_size, device)
        self._wx = self._area_weights(width, output_size, device)
        n = torch.arange(output_size, device=device, dtype=torch.float32)
        k = torch.arange(self.size, device=device, dtype=torch.float32)[:, None]
        dct = torch.cos(math.pi / output_size * (n[None, :] + 0.5) * k)
        dct[0] *= math.sqrt(1.0 / output_size)
        if self.size > 1:
            dct[1:] *= math.sqrt(2.0 / output_size)
        self._dct = dct
        self._shape = shape

    def hash_frame(self, frame_tensor):
        # Reproduce OpenCV's integer BGR->GRAY result from our RGB CUDA tensor.
        frame = frame_tensor[0, :, :h, :w].float().mul(255.0).round_()
        gray = torch.floor(
            (
                frame[0] * 4899.0
                + frame[1] * 9617.0
                + frame[2] * 1868.0
                + 8192.0
            )
            / 16384.0
        )
        self._prepare(gray.shape[0], gray.shape[1], gray.device)
        resized = self._wy @ gray @ self._wx.t()
        resized = resized.round_()
        max_value = resized.max().clamp_min(1.0)
        resized = resized / max_value
        low = self._dct @ resized @ self._dct.t()
        flat = low.flatten().sort().values
        middle = flat.numel() // 2
        median = (flat[middle - 1] + flat[middle]) * 0.5
        return low > median

    def process_frame(self, frame_tensor, frame_index):
        curr_hash = self.hash_frame(frame_tensor)
        self.last_hash_cpu = curr_hash.byte().cpu().numpy().astype(bool)
        if self.last_hash is None:
            self.last_hash = curr_hash
            self.last_score = None
            return False
        hash_dist = torch.count_nonzero(curr_hash != self.last_hash).item()
        hash_dist_norm = hash_dist / self.size_sq
        self.last_hash = curr_hash
        self.last_score = hash_dist_norm
        if (
            hash_dist_norm >= self.threshold
            and frame_index - self.last_scene_cut >= self.min_scene_len_frames
        ):
            self.last_scene_cut = frame_index
            return True
        return False

def detect_scenes_ffmpeg():
    pattern = re.compile(
        r"showinfo.*?\bn:\s*(\d+).*?\bpts:\s*(\d+).*?\bpts_time:\s*([0-9.]+)",
        re.IGNORECASE
    )

    def parse_showinfo(stderr):
        # Match lines containing showinfo with 'n:', 'pts:', and 'pts_time:'
        for line in stderr.splitlines():
            m = pattern.search(line)    
            if m:
                n = int(m.group(1))
                pts = int(m.group(2))
                pts_time = float(m.group(3))
                yield n, pts, pts_time

    print("Running stupid 2-pass scene detection...")

    scene_vid = args.scene_video if args.scene_video else args.video
    # 1st pass: full showinfo to map pts -> frame
    out1 = subprocess.run(
        ["ffmpeg", "-i", scene_vid, "-vf", "showinfo", "-f", "null", "-", "-hide_banner"],
        stderr=subprocess.PIPE, text=True
    )
    pts_to_frame = {pts: frame for frame, pts, _ in parse_showinfo(out1.stderr)}

    # 2nd pass: scene-detected pts_times
    out2 = subprocess.run(
        ["ffmpeg", "-i", scene_vid, "-vf", "select='gt(scene,0.15)',showinfo", "-f", "null", "-", "-hide_banner"],
        stderr=subprocess.PIPE, text=True
    )
    
    # Sort detected scenes by pts_time and debounce by 0.3s
    raw_scenes = list(parse_showinfo(out2.stderr))
    raw_scenes.sort(key=lambda x: x[2])  # sort by pts_time
    
    scene_changes = set()
    last_t = -999.0
    for _, pts, pts_time in raw_scenes:
        if pts in pts_to_frame:
            if pts_time - last_t >= 0.3:
                scene_changes.add(math.ceil(pts_to_frame[pts] / args.drop_input))
                last_t = pts_time

    return scene_changes

scene_changes = set()
gpu_scene_detector = None

if args.scene_detector == 'ffmpeg':
    scene_changes = detect_scenes_ffmpeg()
    print(f"Detected {len(scene_changes)} scene changes via ffmpeg.\n{scene_changes}")
elif args.scene_detector == 'hash':
    min_scene_len_frames = max(1, int(round(args.scene_min_len * source_fps)))
    detector_options = dict(
        threshold=args.scene_hash_threshold,
        size=args.scene_hash_size,
        lowpass=args.scene_hash_lowpass,
        min_scene_len_frames=min_scene_len_frames,
    )
    gpu_scene_detector = TorchHashSceneDetector(**detector_options)
    hash_res = args.scene_hash_size * args.scene_hash_lowpass
    print(
        "Using live hash scene detection on decoded frames "
        f"({hash_res}x{hash_res} DCT input, threshold={args.scene_hash_threshold}, "
        f"min_gap={min_scene_len_frames} kept frames)."
    )
else:
    print("Scene detection disabled.")

vid_out_name = None
if args.output is not None:
    vid_out_name = args.output
else:
    multi = int(args.fps / source_fps) if source_fps else 1
    vid_out_name = '{}_{}X_{}fps.{}'.format(
        video_path_wo_ext, multi, int(np.round(args.fps)), args.ext
    )

print(f"Output Video Name: {vid_out_name}")
print(f"Using persistent CUDA-direct TorchCodec/{'av1_nvenc' if args.use_av1 else 'h264_nvenc'} output.")
from torchcodec.encoders import Encoder

maxrate = int(0.227 * w * h * args.fps)
gop = args.gop if args.frame_range is not None and args.out_chunks else round(args.fps * 2)
nvenc_options = {
    "rc": "1",  # vbr
    "cq": "24",
    "maxrate": str((maxrate // 1_000_000) * 1_000_000),
    "bufsize": str(((maxrate * 2) // 1_000_000) * 1_000_000),
    "rc-lookahead": "48",
    "spatial_aq": "1",
    "temporal_aq": "1",
    "aq-strength": "10",
    "bf": "3",
    "refs": "4",
    "g": str(gop),
    "b": "0",
    "tune": "1",  # hq
}
if not args.use_av1:
    nvenc_options["profile"] = "2"  # h264_nvenc: high
if args.frame_range is not None and args.out_chunks:
    nvenc_options["forced-idr"] = "1"

encoded_path = f"{vid_out_name}.gpu-assembled.mp4" if args.out_chunks else vid_out_name
video_encoder = Encoder()
video_stream = video_encoder.add_video(
    height=h,
    width=w,
    frame_rate=args.fps,
    device="cuda",
    codec="av1_nvenc" if args.use_av1 else "h264_nvenc",
    preset=16,  # h264_nvenc/av1_nvenc: p5
    extra_options=nvenc_options,
)
video_encoder.open_file(encoded_path)
encode_buffer = []

def encode_frame(frame):
    encode_buffer.append(frame)
    if len(encode_buffer) == 8:
        flush_encoder()

def flush_encoder():
    if encode_buffer:
        torch.cuda.synchronize()
        video_stream.add_frames(torch.stack(encode_buffer))
        encode_buffer.clear()

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        frame_index = 0
        while True:
            frame = videogen.read()
            if frame is None:
                break
            if args.fixed_height is not None:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            if frame_index % user_args.drop_input == 0:
                read_buffer.put(frame)
            frame_index += 1
    except:
        pass
    read_buffer.put(None)

def pad_image(img):
    return F.pad(img, padding).half().contiguous()

def frame_to_tensor(frame):
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    img = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
    img = img.to(device, non_blocking=True).unsqueeze(0).float() / 255.0
    return pad_image(img)

scale = 1

tmp = max(args.pad_multiple, int(args.pad_multiple / scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frames)
read_buffer = Queue(maxsize=125)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))

I0 = None
I1 = frame_to_tensor(lastframe)
if gpu_scene_detector is not None:
    gpu_scene_detector.process_frame(I1, 0)
temp = None # save lastframe when processing static frame
time = float(decode_start)
output_index = (
    decode_start * ratio.numerator // ratio.denominator
    if args.frame_range is not None
    else 0
)
emitted_frames = 0

idx_curr = decode_start
idx_prev_unique = decode_start
idx_last_unique = decode_start
idx_prev_prev_unique = max(0, decode_start - 1)

last_unique_frame_64 = cv2.resize(cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA) if args.dedup else None
dedup_skipped = 0

def draw_debug_visual(frame, idx_prev_unique, idx_last_unique, d, time_val, frame_type):
    """
    Draw debug visualization with shape-based indicators
    frame_type: 'interp', 'source', or 'copy'
    """
    
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]

    next_scene_change = None
    if scene_changes:
        # Find smallest scene change >= current idx_last_unique
        future_changes = [sc for sc in scene_changes if sc >= idx_last_unique]
        if future_changes:
            next_scene_change = min(future_changes)
    
    # Visual parameters
    color = (0, 255, 0)  # Green for all
    thickness = 2
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    font_thickness = 2
    margin = 200
    timeline_w = 400
    timeline_y = 100
    marker_size = 10  # Base size for markers
    
    # Position elements in top-right
    x_start = w - timeline_w - margin
    x_end = w - margin
    
    # Draw next scene change info
    sc_text = f"Next SC: {next_scene_change}" if next_scene_change is not None else "No scene change"
    cv2.putText(frame, sc_text, (x_start, timeline_y + 45), 
                font, font_scale*0.8, (0, 200, 255), font_thickness)
    
    # Draw timeline base
    cv2.line(frame, (x_start, timeline_y), (x_end, timeline_y), (100, 100, 100), thickness)
    
    # Calculate current position
    x_current = int(x_start + d * timeline_w)
    label = f"{time_val:.2f}"
    which_side = None
    
    # Draw current position marker with different shapes
    if frame_type == 'interp':
        # Circle for interpolated frames
        cv2.circle(frame, (x_current, timeline_y), marker_size, color, -1)
    elif frame_type == 'source':
        # Square for source frames
        top_left = (x_current - marker_size, timeline_y - marker_size)
        bottom_right = (x_current + marker_size, timeline_y + marker_size)
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        which_side = 0 if d < 0.5 else 1
    else:  # copy or extra
        # Triangle for copied frames
        pts = np.array([
            [x_current, timeline_y - marker_size],  # Top point
            [x_current - marker_size, timeline_y + marker_size],  # Bottom left
            [x_current + marker_size, timeline_y + marker_size]  # Bottom right
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], color)
    
    # Draw label above current position
    text_size = cv2.getTextSize(label, font, font_scale*0.9, font_thickness)[0]
    text_x = x_current - text_size[0] // 2
    cv2.putText(frame, label, (text_x, timeline_y - 20), 
                font, font_scale*0.9, color, font_thickness)
    if which_side != 0:
        cv2.circle(frame, (x_start, timeline_y), marker_size, (200, 200, 200), thickness)
        cv2.putText(frame, f"{idx_prev_unique}", (x_start-15, timeline_y-20), 
                    font, font_scale*0.9, (200, 200, 200), font_thickness)
    if which_side != 1:
        cv2.circle(frame, (x_end, timeline_y), marker_size, (200, 200, 200), thickness)
        cv2.putText(frame, f"{idx_last_unique}", (x_end-25, timeline_y-20), 
            font, font_scale*0.9, (200, 200, 200), font_thickness)
    return frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
        
    is_eof = False
    if frame is None:
        decoded_all_frames = idx_curr + 1 == int(round(tot_frames))
        if (
            range_end_output is not None
            and range_end == int(round(tot_frames))
            and decoded_all_frames
            and output_index < range_end_output
        ):
            I0 = I1
            idx_prev_unique = idx_last_unique
            idx_last_unique = range_end
            is_eof = True
        elif args.dedup and time <= idx_curr:
            I1 = I0
            idx_last_unique = idx_curr
            is_eof = True
        else:
            break
    else:
        idx_curr += 1
        next_I1 = None
        if gpu_scene_detector is not None:
            next_I1 = frame_to_tensor(frame)
            gpu_cut = gpu_scene_detector.process_frame(next_I1, idx_curr)
            if gpu_cut:
                scene_changes.add(idx_curr)

        if args.dedup:
            frame_gray_64 = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
            
            if idx_curr not in scene_changes:
                diff_full = np.abs(frame_gray_64.astype(np.float32) - last_unique_frame_64.astype(np.float32))
                global_mad = np.mean(diff_full)
                
                blocks = diff_full.reshape(8, 8, 8, 8)
                max_block_mad = np.max(np.mean(blocks, axis=(1, 3)))
                
                if global_mad < args.dedup_global_thresh and max_block_mad < args.dedup_block_thresh:
                    dedup_skipped += 1
                    continue
                    
            last_unique_frame_64 = frame_gray_64

        idx_prev_prev_unique = idx_prev_unique
        idx_prev_unique = idx_last_unique
        idx_last_unique = idx_curr

        Im1 = I0
        I0 = I1
        if next_I1 is None:
            next_I1 = frame_to_tensor(frame)
        I1 = next_I1

    output = []
    batch_slots = []
    close_enough = 0.0001
    while range_start_output is not None and output_index < range_start_output and time <= idx_last_unique + close_enough:
        output_index += 1
        time = output_index * timestep

    while time <= idx_last_unique + close_enough and (range_end_output is None or output_index < range_end_output):
        d = (time - idx_prev_unique) / (idx_last_unique - idx_prev_unique) if idx_last_unique > idx_prev_unique else 0
        
        if idx_last_unique in scene_changes:
            if Im1 is None:
                res = I0
                frame_type = 'copy'
            else:
                gap_prev = idx_prev_unique - idx_prev_prev_unique
                if gap_prev == 0:
                    gap_prev = 1
                extrap_factor = (time - idx_prev_unique) / gap_prev
                
                res = run_inference(Im1, I0, 1.0 + extrap_factor, scale)
                frame_type = 'extra'
        else:
            if d < close_enough:
                res = I0
                frame_type = 'source'
            elif d > 1 - close_enough:
                res = I1
                frame_type = 'source'
            elif is_eof:
                res = I0
                frame_type = 'copy'
            else:
                res = None
                frame_type = 'interp'
        
        output.append([res, d, frame_type, time])
        if res is None:
            batch_slots.append((len(output) - 1, d))
        output_index += 1
        time = output_index * timestep if range_start_output is not None else time + timestep

    for batch_start in range(0, len(batch_slots), args.batch_timestamps):
        slots = batch_slots[batch_start:batch_start + args.batch_timestamps]
        if len(slots) == 1:
            slot, d = slots[0]
            output[slot][0] = run_inference(I0, I1, d, scale)
        else:
            batch_size = len(slots)
            timesteps = torch.tensor(
                [d for _, d in slots],
                device=I0.device,
                dtype=I0.dtype,
            ).view(batch_size, 1, 1, 1)
            try:
                batched = run_inference(
                    I0.expand(batch_size, -1, -1, -1),
                    I1.expand(batch_size, -1, -1, -1),
                    timesteps,
                    scale,
                )
                for batch_index, (slot, _) in enumerate(slots):
                    output[slot][0] = batched[batch_index:batch_index + 1]
            except torch.cuda.OutOfMemoryError:
                del timesteps
                torch.cuda.empty_cache()
                print(
                    f"Timestamp batch {batch_size} exceeded GPU memory; "
                    "retrying those frames serially."
                )
                for slot, d in slots:
                    output[slot][0] = run_inference(I0, I1, d, scale)

    emitted_frames += len(output)
    for res, d, frame_type, frame_time in output:
        packed = res[0, :, :h, :w].mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
        if args.debug:
            debug_frame = draw_debug_visual(
                packed[[2, 1, 0]].cpu().numpy().transpose(1, 2, 0),
                idx_prev_unique,
                idx_last_unique,
                d,
                frame_time,
                frame_type,
            )
            packed = torch.from_numpy(
                np.ascontiguousarray(debug_frame[:, :, ::-1].transpose(2, 0, 1))
            ).to(device)
        encode_frame(packed.contiguous())
    
    pbar.update(idx_last_unique - idx_prev_unique)
    
    if frame is not None:
        lastframe = frame
        
    if (range_end_output is not None and output_index >= range_end_output) or is_eof:
        break

range_error = None
if args.frame_range is None:
    encode_frame(
        torch.from_numpy(
            np.ascontiguousarray(lastframe[:, :, ::-1].transpose(2, 0, 1))
        ).to(device)
    )
elif emitted_frames != expected_range_frames:
    range_error = RuntimeError(
        f"range produced {emitted_frames} frames; expected {expected_range_frames}. "
        "The END boundary is past the decodable source timeline."
    )
flush_encoder()
video_encoder.close()
if args.out_chunks:
    segment_args = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", encoded_path,
        "-map", "0:v:0", "-c", "copy", "-f", "segment",
        "-reset_timestamps", "1",
    ]
    if args.frame_range is not None:
        segment_args += ["-segment_frames", segment_frames]
    else:
        segment_args += ["-segment_time", "10"]
    output_path = Path(vid_out_name)
    chunk_pattern = vid_out_name if "%" in output_path.name else str(
        output_path.with_name(f"{output_path.stem}_%03d{output_path.suffix}")
    )
    subprocess.run(segment_args + [chunk_pattern], check=True)
    Path(encoded_path).unlink()
pbar.close()
if args.dedup:
    print(f"Dedup: skipped {dedup_skipped} duplicate frames.")
if gpu_scene_detector is not None:
    print(f"Detected {len(scene_changes)} scene changes via live GPU hash.\n{scene_changes}")
if range_error is not None:
    raise range_error
