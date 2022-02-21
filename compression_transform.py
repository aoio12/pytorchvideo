import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
import imageio
import math
import numbers
import random
import warnings
import copy
from collections.abc import Sequence
from typing import Tuple, List, Optional, Dict, Callable

import torch
from torch import Tensor

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)

# try:
#     import accimage
# except ImportError:
#     accimage = None

import pytorchvideo.transforms.functional
from torchvision.transforms import Compose
from pytorchvideo.transforms import Normalize
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
import torchvision.transforms
from pytorchvideo.data import ClipSampler
from typing import Any, Dict, NamedTuple, Optional, Tuple


class JpegCompression():
    def __init__(self, compression, config):
        self.compression = compression
        self.config = config

    def __call__(self, img, orig_img):
        # args = Args()
        aug = iaa.JpegCompression(self.compression)
        for i in range(int(self.config['VIDEO_NUM_SUBSAMPLED'])):
            # image = img[:, i, :, :].permute(1, 2, 0)
            # # image = image.to('cpu').detach().numpy().copy()
            # image = image.numpy()
            # image = image * 255
            # # image = image.numpy()
            # image = image.clip(0, 255).astype(np.uint8)
            # # image = image.astype(np.uint8)
            # tmp = aug.augment_image(image)
            # tmp = np.asarray(tmp).astype('float32')
            # img[:, i, :, :] = torch.from_numpy(tmp).permute(2, 0, 1)
            # # img[:,i,:,:] = torch.from_numpy(tmp).clone().permute(2,0,1)
            # # orig_img = orig_img / 255
            image = img[:, i, :, :].permute(1, 2, 0)
            image = image.to('cpu').detach().numpy().copy()
            image = image * 255
            image = image.clip(0, 255).astype(np.uint8)
            image = aug.augment_image(image)
            tmp = np.asarray(image)
            tmp = tmp.astype('float32')
            tmp = tmp / 255
            img[:, i, :, :] = torch.from_numpy(tmp).clone().permute(2, 0, 1)
            original = orig_img[:, i, :, :].permute(1, 2, 0)
            original = original / 255
            orig_img[:, i, :, :] = original.permute(2, 0, 1)
        return img, orig_img


class CompCompose(Compose):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        orig_img = copy.deepcopy(img)
        for t in self.transforms:
            img, orig_img = t(img, orig_img)
        return img, orig_img


class BothCompose(Compose):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, orig_img):
        for t in self.transforms:
            img, orig_img = t(img, orig_img)
        return img, orig_img


class CompApplyTransformToKey(ApplyTransformToKey):

    def __init__(self, key: str, transform: Callable):
        super().__init__(key, transform)

    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]):
        x[self._key], y[self._key] = self._transform(
            x[self._key], y[self._key])
        return x, y


class CompUniformTemporalSubsample(UniformTemporalSubsample):

    def __init__(self, num_samples: int):
        super().__init__(num_samples)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = pytorchvideo.transforms.functional.uniform_temporal_subsample(
            x, self._num_samples),
        y = pytorchvideo.transforms.functional.uniform_temporal_subsample(
            y, self._num_samples)
        return x, y


class CompNormalize(torchvision.transforms.Normalize):

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # vid_x = x.permute(1, 0, 2, 3)
        # vid_x = super().forward(vid_x)
        # vid_x = vid_x.permute(1, 0, 2, 3)

        vid_y = y.permute(1, 0, 2, 3)
        vid_y = super().forward(vid_y)
        vid_y = vid_y.permute(1, 0, 2, 3)

        return x, vid_y


class CompRandomShortSideScale(RandomShortSideScale):
    def __init__(self, min_size: int, max_size: int):
        super().__init__(min_size, max_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        x = pytorchvideo.transforms.functional.short_side_scale(x, size)
        y = pytorchvideo.transforms.functional.short_side_scale(y, size)
        return x, y


class CompShortSideScale(ShortSideScale):
    def __init__(self, size: int):
        super().__init__(size)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = pytorchvideo.transforms.functional.short_side_scale(x, self._size)
        y = pytorchvideo.transforms.functional.short_side_scale(y, self._size)
        return x, y


class CompRandomCrop(RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding=padding, pad_if_needed=pad_if_needed,
                         fill=fill, padding_mode=padding_mode)

    def forward(self, img, orig_img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(orig_img, i, j, h, w)


class CompCenterCrop(CenterCrop):
    def __init__(self, size):
        super().__init__(size)

    def forward(self, img, orig_img):
        return F.center_crop(img, self.size), F.center_crop(orig_img, self.size)


class CompRandomHorizontalFlip(RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p=p)

    def forward(self, img, orig_img):

        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(orig_img)
        return img, orig_img


class CompRemoveKey(RemoveKey):
    def __init__(self, key: str):
        super().__init__(key)

    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]):
        if self._key in x:
            del x[self._key]
        if self._key in y:
            del y[self._key]

        return x, y


class CompLambda(Lambda):
    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, img, orig_img):
        img = img[0]
        # orig_img = orig_img[0]
        return self.lambd(img), orig_img
        # return self.lambd(img), self.lambd(orig_img)


class BothLambda(Lambda):
    def __init__(self, lambd):
        super().__init__(lambd)

    def __call__(self, img, orig_img):
        return self.lambd(img), self.lambd(orig_img)


class ClipInfo(NamedTuple):
    """
    Named-tuple for clip information with:
        clip_start_sec  (float): clip start time.
        clip_end_sec (float): clip end time.
        clip_index (int): clip index in the video.
        aug_index (int): augmentation index for the clip. Different augmentation methods
            might generate multiple views for the same clip.
        is_last_clip (bool): a bool specifying whether there are more clips to be
            sampled from the video.
    """

    clip_start_sec: float
    clip_end_sec: float
    clip_index: int
    aug_index: int
    is_last_clip: bool


class SameClipSampler(ClipSampler):
    """
    Randomly samples clip of size clip_duration from the videos.
    """

    def __call__(
            self, last_clip_time: float, video_duration: float, annotation: Dict[str, Any]) -> ClipInfo:
        """
        Args:
            last_clip_time (float): Not used for RandomClipSampler.
            video_duration: (float): the duration (in seconds) for the video that's
                being sampled
            annotation (Dict): Not used by this sampler.
        Returns:
            clip_info (ClipInfo): includes the clip information of (clip_start_time,
            clip_end_time, clip_index, aug_index, is_last_clip). The times are in seconds.
            clip_index, aux_index and is_last_clip are always 0, 0 and True, respectively.
        """
        # max_possible_clip_start = max(video_duration - self._clip_duration, 0)
        clip_start_sec = 0
        return ClipInfo(
            clip_start_sec, clip_start_sec + self._clip_duration, 0, 0, True
        )
