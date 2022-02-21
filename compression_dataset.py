from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.video import VideoPathHandler

from pytorchvideo.data.labeled_video_dataset import LabeledVideoPaths
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.data.utils import MultiProcessSampler

logger = logging.getLogger(__name__)


class SegLabeledVideoDataset(LabeledVideoDataset):

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decoder: str = "pyav"
    ) -> None:

        self._decode_audio = decode_audio
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        self._decoder = decoder

        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._next_clip_start_time = 0.0
        self.video_path_handler = VideoPathHandler()

    def __next__(self):

        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(
                MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            # Only load the clip once and reuse previously stored clip if there are multiple
            # views for augmentations to perform on the same clip.
            if aug_index == 0:
                self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._next_clip_start_time = clip_end

            video_is_null = (
                self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if is_last_clip or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(
                            video.name, i_try)
                    )
                    continue

            frames = self._loaded_clip["video"]
            audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict, row_video = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict, row_video
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )


def segUcf101(
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = True,
        decoder: str = "pyav"):

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Ucf101")

    return seg_labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )


def seg_labeled_video_dataset(
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = True,
        decoder: str = "pyav"):

    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = SegLabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )

    return dataset


def segkinetics(
    data_path: str,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = True,
    decoder: str = "pyav",
) -> SegLabeledVideoDataset:

    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Kinetics")

    return seg_labeled_video_dataset(
        data_path,
        clip_sampler,
        video_sampler,
        transform,
        video_path_prefix,
        decode_audio,
        decoder,
    )
