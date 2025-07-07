# Smoothie 🍹

> Silky-smooth slow-mo from any video, no blender needed.

---

## Installation

```bash
git clone https://github.com/yourusername/smoothie.git
cd smoothie
pip install -r requirements.txt
```

Download or train your RIFE HD model and place weights in `train_log/`.

---

## Usage

```bash
python interpolate_video.py \
  --video input.mp4       # Path to source video
  --fps 60                # Target FPS (e.g., 24, 30, 60, 240)
  --output out.mp4        # (Optional) Output file path
  --model train_log       # Model directory
  --fp16                  # (Optional) Enable FP16
  --png                   # (Optional) Export PNG frames
  --fixed_height 1440      # (Optional) Downscale height
```

Run `python interpolate_video.py --help` for full options.

---

## License

MIT © 2025
