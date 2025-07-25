"""
python inference.py \
    --variant "mobilenetv3" \
    --checkpoint "checkpoint/rvm_mobilenetv3.pth" \
    --device "cuda" \
    --input-source "video/input.mp4" \
    --output-type "video" \
    --output-background "green" \
    --output-composition "video/composition.mp4" \
    --output-alpha "video/alpha.mp4" \
    --output-foreground "video/foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1 \
    --num-workers 1
"""

import av
import argparse
import torch
import os
from PIL import Image
from .model import MattingNetwork
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from .inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter, AudioVideoWriter


def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_background: Optional[str] = 'default',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  require_audio: bool = True,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):

    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_background: Options: ["default", "green", "white", "image"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        require_audio: Keep audio from the input video.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """

    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert output_background in ['default', 'green', 'white', 'image'], 'Support "default" "green" "white" and "image" backgrounds. "video" default is green, "png_sequence" default is transparent.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)

    if output_type == 'video' and require_audio and os.path.isfile(input_source):
        try:
            container = av.open(input_source)
            if container.streams.get(audio=0):
                audio_source = container.streams.get(audio=0)[0]
        except:
            audio_source = None

    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if require_audio and audio_source:
            if output_composition is not None:
                writer_com = AudioVideoWriter(
                    path=output_composition,
                    frame_rate=frame_rate,
                    audio_stream=audio_source,
                    bit_rate=int(output_video_mbps * 1000000))
        else:
            if output_composition is not None:
                writer_com = VideoWriter(
                    path=output_composition,
                    frame_rate=frame_rate,
                    bit_rate=int(output_video_mbps * 1000000))

        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))

        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    if output_composition is not None:
        if output_background == 'image':
            bgr = None
        elif output_background == 'default' and output_type == 'png_sequence':
            bgr = None
        elif output_background == 'white':
            bgr = torch.tensor([255, 255, 255], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)  # white background
        else:
            bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)  # green background

    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for src in reader:
                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_background == 'image':
                        # print(src.shape)
                        h, w = src.shape[3:]
                        loader = transforms.Compose([
                            transforms.Resize(size=(h, w)),
                            transforms.ToTensor()
                        ])
                        img = Image.open("work/background/background.jpg")
                        bgr = loader(img).to(device, dtype, non_blocking=True)
                        com = fgr * pha + bgr * (1 - pha)
                    elif output_background == 'default' and output_type == 'png_sequence':
                        # transparent
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    else:
                        com = fgr * pha + bgr * (1 - pha)

                    writer_com.write(com[0])
                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device

    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--input-source', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str)
    parser.add_argument('--output-alpha', type=str)
    parser.add_argument('--output-foreground', type=str)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-background', type=str, choices=['default', 'green', 'white', 'image'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--require-audio', type=bool, default=True)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    args = parser.parse_args()

    converter = Converter(args.variant, args.checkpoint, args.device)
    converter.convert(
        input_source=args.input_source,
        input_resize=args.input_resize,
        downsample_ratio=args.downsample_ratio,
        output_type=args.output_type,
        output_background=args.output_background,
        output_composition=args.output_composition,
        output_alpha=args.output_alpha,
        output_foreground=args.output_foreground,
        output_video_mbps=args.output_video_mbps,
        require_audio=args.require_audio,
        seq_chunk=args.seq_chunk,
        num_workers=args.num_workers,
        progress=not args.disable_progress
    )