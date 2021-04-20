from typing import List

from torchvision import transforms
from solvers.rgb_to_lab import _tensor_to_rgb_, lab_denormalize
from solvers.solver import StarGANv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import ffmpeg
from torch import Tensor


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(gan: StarGANv2, x_src: Tensor, s_prev: Tensor, s_next: Tensor):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    alphas = get_alphas()
    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = gan.generate_image_with_style(x_src, s_ref)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(
            entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=0):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas)  # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


def tensor2ndarray255(images):
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255


@torch.no_grad()
def video_ref(gan: StarGANv2, x_src: Tensor, x_ref: Tensor, y_ref: Tensor, fname: str):
    video = []
    s_ref = gan.generate_image_style(x_ref, y_ref)
    s_prev = None

    N = x_ref.shape[0]
    for i in range(N):
        x_next, y_next, s_next = x_ref[i:i+1], y_ref[i:i+1], s_ref[i:i+1]
        if s_prev is None:
            x_prev, s_prev = x_next, s_next
            continue

        interpolated = interpolate(gan, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        # (T, C, 256*2, 256*(batch+1))
        frames = torch.cat([slided, interpolated], dim=3).cpu()
        video.append(frames)
        x_prev, s_prev = x_next, s_next

    video = torch.cat(video)
    video = _tensor_to_rgb_(video)
    video = tensor2ndarray255(video)

    save_video(fname, video)


@torch.no_grad()
def video_latent(gan: StarGANv2, x_src: Tensor, s_list: List[Tensor], fname: str):
    s_prev = None
    video = []
    # fetch reference images
    for s_next in s_list:
        s_next = s_next.repeat(1, 1)
        if s_prev is None:
            s_prev = s_next
            continue
        frames = interpolate(gan, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next

    video = torch.cat(video)
    print(video.shape)
    video = _tensor_to_rgb_(video)
    video = tensor2ndarray255(video)

    save_video(fname, video)


def save_video(fname: str, images: Tensor, output_fps=30, vcodec='libx264', filters=''):
    # assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo',
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    # 2*PTS is for slower playback
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p',
                           vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()
