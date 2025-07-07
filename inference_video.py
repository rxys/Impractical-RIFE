import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
from queue import Queue, Empty
from vidgear.gears import WriteGear
from vidgear.gears import VideoGear

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None, required=True)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--fps', dest='fps', type=float, default=None, required=True)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--drop_input', dest='drop_input', type=int, default=1, help='Only keep every Nth input frame (1 = keep all, 2 = drop every other, etc.)')
parser.add_argument('--fixed_height', type=int, default=None, help='Fixed vertical resolution for downscaling while keeping aspect ratio')

args = parser.parse_args()

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

# CHANGED: Run ffmpeg to get scene change frame indices on demand
print("Running ffmpeg scene detection...")
scene_changes = set()
ffmpeg_cmd = [
    "ffmpeg", "-i", args.video,
    "-vf", "select='gt(scene,0.4)',showinfo",
    "-f", "null", "-"
]
result = subprocess.run(ffmpeg_cmd, stderr=subprocess.PIPE, text=True)
for line in result.stderr.splitlines():
    if "showinfo" in line and "pts_time" in line:
        if "n:" in line:
            parts = line.split("n:")
            try:
                frame_num = int(parts[1].split()[0])
                scene_changes.add(frame_num - 1)  # CHANGED: shift to duplicate *before* scene change
            except:
                continue
print(f"Detected {len(scene_changes)} scene changes via ffmpeg.")

vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    output_params = {
        "-input_framerate": args.fps,
        "-vcodec": "h264_nvenc",
        "-rc": "vbr",
        "-cq": "22",
        "-maxrate": "50M",
        "-bufsize": "100M",          # 2x maxrate
        "-preset": "p5",
        "-rc-lookahead": "48",
        "-spatial_aq": "1",
        "-temporal_aq": "1",
        "-aq-strength": "10",
        "-bf": "3",
        "-refs": "4",
        "-g": "120",
        "-profile:v": "high",
        "-pix_fmt": "yuv420p",
        "-b:v": "0",
        # Only if Turing/Ampere GPU:
        "-tune": "hq",
        "-weighted_pred": "1"
    }
    
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
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

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None # save lastframe when processing static frame
time = 0
n = 0

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)

    output = []
    close_enough = 0.0001
    while time + timestep <= n + 1 + close_enough:
        if (n + 1) in scene_changes:
            output.append(I0)
        else:
            assert model.version >= 3.9
            d = time - n
            if d < close_enough:
                res = I0
            elif d > 1 - close_enough:
                res = I1
            else:
                res = model.inference(I0, I1, d, scale)
            output.append(res)
        time += timestep

    for mid in output:
        mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
        write_buffer.put(mid[:h, :w])
    pbar.update(1)
    n += 1
    lastframe = frame

write_buffer.put(lastframe)
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.close()
