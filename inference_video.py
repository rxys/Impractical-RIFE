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
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--fps', dest='fps', type=float, default=None, required=True)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--drop_input', dest='drop_input', type=int, default=1, help='Only keep every Nth input frame (1 = keep all, 2 = drop every other, etc.)')
parser.add_argument('--fixed_height', type=int, default=None, help='Fixed vertical resolution for downscaling while keeping aspect ratio')
parser.add_argument('--debug', dest='debug', action='store_true', help='Enable debug visualization')

args = parser.parse_args()

from model.warplayer import warp

def forward_monkey(self, x, timestep=0.5, scale_list=[16, 8, 4, 2, 1], training=False, fastmode=True, ensemble=False):
    if training == False:
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

    # Extrapolation for timestep > 1
    if not training and isinstance(timestep, float) and timestep > 1.0:
        # Step 1: Get flow from img0 to img1 (at timestep=1)
        flow_list, _, merged = self.forward(x, timestep=1.0, scale_list=scale_list, training=False, fastmode=True, ensemble=False)
        
        # Step 2: Compute scaled flow for timestep > 1
        flow_0_to_1 = flow_list[-1][:, :2]  # Final flow from img0 to img1
        flow_extrapolate = (timestep - 1.0) * flow_0_to_1
        
        # Step 3: Warp img1 using scaled flow
        extrapolated_frame = warp(img1, flow_extrapolate)
        
        # Replace last merged entry with extrapolated frame
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

def detect_scenes():
    pattern = re.compile(
        r"showinfo.*?\bn:\s*(\d+).*?\bpts:\s*(\d+)",
        re.IGNORECASE
    )

    def parse_showinfo(stderr):
        # Match lines containing showinfo with 'n:' and 'pts_time:'
        for line in stderr.splitlines():
            m = pattern.search(line)    
            if m:
                n = int(m.group(1))
                pts = int(m.group(2))
                yield n, pts

    print("Running stupid 2-pass scene detection...")

    # 1st pass: full showinfo to map pts_time -> frame
    out1 = subprocess.run(
        ["ffmpeg", "-i", args.video, "-vf", "showinfo", "-f", "null", "-", "-hide_banner"],
        stderr=subprocess.PIPE, text=True
    )
    pts_to_frame = {pts: frame for frame, pts in parse_showinfo(out1.stderr)}

    # 2nd pass: scene-detected pts_times
    out2 = subprocess.run(
        ["ffmpeg", "-i", args.video, "-vf", "select='gt(scene,0.4)',showinfo", "-f", "null", "-", "-hide_banner"],
        stderr=subprocess.PIPE, text=True
    )
    scene_changes = {
        math.ceil(pts_to_frame[pts] / args.drop_input)
        for _, pts in parse_showinfo(out2.stderr)
        if pts in pts_to_frame
    }
    return scene_changes

scene_changes = detect_scenes()
print(f"Detected {len(scene_changes)} scene changes via ffmpeg.\n{scene_changes}")

vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    max_bpp = 0.227  # Targeted to match 50 Mbps at 1440p60
    maxrate = int(max_bpp * w * h * args.fps)
    output_params = {
        "-input_framerate": args.fps,
        "-vcodec": "h264_nvenc",
        "-rc": "vbr",
        "-cq": "24",
        "-maxrate": f"{maxrate // 1_000_000}M",
        "-bufsize": f"{(maxrate * 2) // 1_000_000}M", # 2x maxrate
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
if not vid_out is None:
    vid_out.close()
