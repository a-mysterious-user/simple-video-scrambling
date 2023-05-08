# assess the quality of video sequences

from video_scramble.video_handler import VideoFrameIterator
import numpy as np
import skimage, skimage.metrics
from typing import Callable


def _compute_metric(vid1:VideoFrameIterator, vid2:VideoFrameIterator, metric_func:Callable, metric_func_kwargs:dict):

    metric_vals = []

    while True:
        try:
            f1 = next(vid1)
            f2 = next(vid2)
        except StopIteration:
            break

        f1 = skimage.img_as_ubyte(f1)
        f2 = skimage.img_as_ubyte(f2)
        metric_vals.append(metric_func(f1, f2, **metric_func_kwargs))

    return np.asarray(metric_vals)


def compute_psnr(vid1:VideoFrameIterator, vid2:VideoFrameIterator, **kwargs):
    return _compute_metric(vid1, vid2, skimage.metrics.peak_signal_noise_ratio, metric_func_kwargs=kwargs)

def compute_ssim(vid1:VideoFrameIterator, vid2:VideoFrameIterator, **kwargs):
    return _compute_metric(vid1, vid2, skimage.metrics.structural_similarity, metric_func_kwargs=kwargs)


