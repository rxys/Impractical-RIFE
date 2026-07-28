# Smoothie 🍹

> Silky-smooth slow-mo from any video, no blender needed.

## Installation

Smoothie needs Linux, an NVIDIA graphics card, and FFmpeg.

```bash
git clone https://github.com/rxys/Impractical-RIFE.git
cd Impractical-RIFE
pip install -r requirements.txt
vapoursynth config
```

The first video may take longer while Smoothie prepares the model. Later runs
can reuse it.

## Usage

```bash
python inference_video.py --video input.mp4 --fps 60 --output smooth.mp4
```

This turns `input.mp4` into a 60 FPS video named `smooth.mp4`.

For more choices:

```bash
python inference_video.py --help
```

## License

MIT © 2025
