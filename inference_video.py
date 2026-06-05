from model.loss import img1
import re
import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import subprocess
from queue import Queue, Empty
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear
import math

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
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--fps', dest='fps', type=float, default=None, required=True)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--drop_input', dest='drop_input', type=int, default=1, help='Only keep every Nth input frame (1 = keep all, 2 = drop every other, etc.)')
parser.add_argument('--fixed_height', type=int, default=None, help='Fixed vertical resolution for downscaling while keeping aspect ratio')
parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug visualization')
parser.add_argument('--av1', dest='use_av1', action='store_true', help='Use software AV1 encoding (libaom-av1) instead of h264_nvenc.')
parser.add_argument('--out_chunks', dest='out_chunks', action='store_true', help='Output streamable chunks via segment muxer')

args = parser.parse_args()

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
    fallback = warp(img, -flow.to(img.dtype))

    return torch.where(weight_sum > 1e-6, result, fallback)


def forward_monkey(self, x, timestep=0.5, scale_list=[16, 8, 4, 2, 1], training=False, fastmode=True, ensemble=False):
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
    else:
        timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
    f0 = self.encode(img0[:, :3])
    f1 = self.encode(img1[:, :3])
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


from train_log.IFNet_HDv3 import IFNet
IFNet.forward = forward_monkey

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if(args.fp16):
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

videoCapture = cv2.VideoCapture(args.video)
source_fps = videoCapture.get(cv2.CAP_PROP_FPS) / args.drop_input
timestep = source_fps / args.fps
tot_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
videoCapture.release()
videogen = VideoGear(source=args.video, backend='ffmpeg').start()
first_frame = videogen.read()
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
live_scene_detector = None

if args.scene_detector == 'ffmpeg':
    scene_changes = detect_scenes_ffmpeg()
    print(f"Detected {len(scene_changes)} scene changes via ffmpeg.\n{scene_changes}")
elif args.scene_detector == 'hash':
    min_scene_len_frames = max(1, int(round(args.scene_min_len * source_fps)))
    live_scene_detector = HashSceneDetector(
        threshold=args.scene_hash_threshold,
        size=args.scene_hash_size,
        lowpass=args.scene_hash_lowpass,
        min_scene_len_frames=min_scene_len_frames,
    )
    live_scene_detector.process_frame(lastframe, 0)
    hash_res = args.scene_hash_size * args.scene_hash_lowpass
    print(
        "Using live hash scene detection on decoded frames "
        f"({hash_res}x{hash_res} DCT input, threshold={args.scene_hash_threshold}, "
        f"min_gap={min_scene_len_frames} kept frames)."
    )
else:
    print("Scene detection disabled.")

vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.use_av1:
        print("Using software AV1 (libaom-av1) encoder for high quality output.")
        # High quality, single-pass CRF settings for libaom-av1
        output_params = {
            "-input_framerate": args.fps,
            "-vcodec": "libaom-av1",
            "-crf": "24", # Target Constant Quality mode
            "-cpu-used": "4", # Balance of speed and quality (Lower is slower/better)
            "-row-mt": "1", # Enable row-based multithreading
            "-pix_fmt": "yuv420p",
            "-b:v": "0", # Ensures CRF mode
            "-tune": "ssim", # Tune for structural similarity/visual quality
        }
    else:
        print("Using hardware H.264 (h264_nvenc).")
        max_bpp = 0.227 
        maxrate = int(max_bpp * w * h * args.fps)
        
        output_params = {
            "-input_framerate": args.fps,
            "-vcodec": "h264_nvenc",
            "-rc": "vbr",
            "-cq": "24",
            "-maxrate": f"{maxrate // 1_000_000}M",
            "-bufsize": f"{(maxrate * 2) // 1_000_000}M",
            "-preset": "p5",
            "-rc-lookahead": "48",
            "-spatial_aq": "1",
            "-temporal_aq": "1",
            "-aq-strength": "10",
            "-bf": "3",
            "-refs": "4",
            "-g": args.fps * 2,
            "-profile:v": "high",
            "-pix_fmt": "yuv420p",
            "-b:v": "0",
            # Only if Turing/Ampere GPU:
            "-tune": "hq",
        }

    if args.out_chunks:
        output_params["-f"] = "segment"
        output_params["-segment_time"] = "10"
        output_params["-reset_timestamps"] = "1"

    if args.output is not None:
        vid_out_name = args.output
    else:
        # Assuming args.multi exists or is derived from args.fps / source_fps
        multi = int(args.fps / source_fps) if source_fps else 1 
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, multi, int(np.round(args.fps)), args.ext)
        
    print(f"Output Video Name: {vid_out_name}")
    # Initialize WriteGear. This can fail if the codec is truly unavailable.
    vid_out = WriteGear(output=vid_out_name, logging=True, **output_params)

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item)  # Write the frame using VidGear
            cnt += 1

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
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

scale = 1

tmp = max(128, int(128 / scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frames)
write_buffer = Queue(maxsize=125)
read_buffer = Queue(maxsize=125)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I0 = None
I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None # save lastframe when processing static frame
time = 0
n = 0

def draw_debug_visual(frame, n, d, frame_type):
    """
    Draw debug visualization with shape-based indicators
    frame_type: 'interp', 'source', or 'copy'
    """
    
    frame = np.ascontiguousarray(frame)
    h, w = frame.shape[:2]

    next_scene_change = None
    if scene_changes:
        # Find smallest scene change >= current n
        future_changes = [sc for sc in scene_changes if sc >= n]
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
    label = f"{n+d:.2f}"
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
    else:  # copy
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
        cv2.putText(frame, f"{n}", (x_start-15, timeline_y-20), 
                    font, font_scale*0.9, (200, 200, 200), font_thickness)
    if which_side != 1:
        cv2.circle(frame, (x_end, timeline_y), marker_size, (200, 200, 200), thickness)
        cv2.putText(frame, f"{n+1}", (x_end-25, timeline_y-20), 
            font, font_scale*0.9, (200, 200, 200), font_thickness)
    return frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    if live_scene_detector is not None and live_scene_detector.process_frame(frame, n + 1):
        scene_changes.add(n + 1)
    Im1 = I0
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    
    output = []
    close_enough = 0.0001
    while time <= n + 1 + close_enough:
        d = time - n
        
        if (n + 1) in scene_changes:
            if Im1 is None:
                res = I0
                frame_type = 'copy'
            else:
                res = model.inference(Im1, I0, 1.0 + d, scale)
                frame_type = 'extra'
        else:
            if d < close_enough:
                res = I0
                frame_type = 'source'
            elif d > 1 - close_enough:
                res = I1
                frame_type = 'source'
            else:
                res = model.inference(I0, I1, d, scale)
                frame_type = 'interp'
        
        output.append((res, d, frame_type))
        time += timestep

    for res, d, frame_type in output:
        mid = ((res[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))
        cropped = mid[:h, :w]
        
        # Add debug visualization
        if args.debug:
            cropped = draw_debug_visual(cropped, n, d, frame_type)
        
        write_buffer.put(cropped)
    
    pbar.update(1)
    n += 1
    lastframe = frame

write_buffer.put(lastframe)
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if live_scene_detector is not None:
    print(f"Detected {len(scene_changes)} scene changes via live hash.\n{scene_changes}")
if not vid_out is None:
    vid_out.close()
