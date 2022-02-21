#sad, ssa, psnr, ssimスコア計算
from comet_ml import Experiment
from ast import parse
from logging import root
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
import numpy as np

from torchvision import transforms

from pytorchvideo.models import x3d
# from pytorchvideo.data import RandomClipSampler, UniformClipSampler
from pytorchvideo.data import Ucf101, RandomClipSampler, UniformClipSampler, Kinetics
# from compression_dataset import segUcf101, segkinetics

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

from tqdm import tqdm
from collections import OrderedDict
import itertools
import os
import pickle

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import argparse
import configparser
from fractions import Fraction


from compression_transform import (
    JpegCompression,
    CompCompose,
    BothCompose,
    CompApplyTransformToKey,
    CompUniformTemporalSubsample,
    CompLambda,
    BothLambda,
    CompNormalize,
    CompRandomShortSideScale,
    CompRandomCrop,
    CompRandomHorizontalFlip,
    CompRemoveKey,
    CompShortSideScale,
    CompCenterCrop,
    SameClipSampler
)

from compression_dataset import segkinetics
import cv2

from img_score import (
    calculate_sad,
    calculate_ssd,
    calculate_psnr,
    calculate_ssim,
    calculate_mae,
    calculate_rmse
)


class LimitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def get_kinetics(subset, compression, root_path, config, en):
    """
    Kinetics400のデータセットを取得
    Args:
        subset (str): "train" or "val"
    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """

    if(en):
        print('圧縮がかかっています')
        transform = CompCompose([
            CompApplyTransformToKey(
                key="video",
                transform=BothCompose([
                    CompUniformTemporalSubsample(
                        int(config['VIDEO_NUM_SUBSAMPLED'])),
                    # print(config['NUM_WORKERS']),
                    # TODO: 受け渡されたjpegcompressionオブジェクトをここに入れる
                    # jpeg_compression,
                    CompLambda(lambda x: x / 255.),
                    JpegCompression(compression, config),
                    # CompNormalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    # RandomShortSideScale(min_size=256, max_size=320,),
                    CompShortSideScale(size=256),
                    CompCenterCrop(224),
                    # RandomHorizontalFlip(),
                ]),
            ),
            CompApplyTransformToKey(
                key="label",
                transform=BothLambda(lambda x: x),
            ),
            CompRemoveKey("audio"),
        ])
    root_kinetics = root_path

    dataset = Kinetics(
        data_path=root_kinetics + subset,
        video_path_prefix=root_kinetics + subset,
        clip_sampler=SameClipSampler(
            clip_duration=float(Fraction(config['CLIP_DURATION']))),
        video_sampler=SequentialSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def make_loader(dataset, config):
    """
    データローダーを作成
    Args:
        dataset (pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset): get_datasetメソッドで取得したdataset
    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    # args = Args()
    loader = DataLoader(LimitDataset(dataset),
                        batch_size=int(config['BATCH_SIZE']),
                        drop_last=True,
                        num_workers=int(config['NUM_WORKERS']))
    return loader


def get_dataset(dataset, subset, compression, root_path, config, en):
    """
    データセットを取得
    Args:
        dataset (str): "Kinetis400" or "UCF101"
        subset (str): "train" or "val"
    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset): 取得したデータセット
    """
    # TODO: get_kineticsにjpegcompressionを渡す
    # print('get_dataset:', jpeg_compression)
    if dataset == "Kinetics400":
        return get_kinetics(subset, compression, root_path, config, en)
    # elif dataset == "UCF101":
    #     return get_ucf101(subset, compression, root_path, config, en)
    return False


# comet debug用


def get_params(config):
    params = {
        "frams_per_clip": int(config['FRAMES_PER_CLIP']),
        "step_between_clips": int(config['STEP_BETWEEN_CLIPS']),
        "batch_size": int(config['BATCH_SIZE']),
        "num_workers": int(config['NUM_WORKERS']),
        "clip_duration": float(Fraction(config['CLIP_DURATION'])),
        "video_num_sabsampled": int(config['VIDEO_NUM_SUBSAMPLED'])
        # "model": config['MODEL']
    }

    return params


def sample_check(compression, root_path, config, en, experiment):
    """学習済みモデルにサンプルデータ100個を流し込んで挙動を確認"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset("Kinetics400", "val",
                          compression, root_path, config, en)

    sample_loader = make_loader(dataset, config)
    print(vars(sample_loader))
    i = 0

    with torch.no_grad():
        with tqdm(enumerate(sample_loader),
                  total=len(sample_loader),
                  leave=True) as pbar_batch:

            for batch_idx, (inputs, original) in pbar_batch:
                # input_list = []
                # print(batch_idx)
                inputs = inputs["video"].to(device)
                original = original["video"].to(device)

                for k in range(int(config['BATCH_SIZE'])):
                    psnr_score_list = []
                    ssim_score_list = []
                    mae_score_list = []
                    rmse_score_list = []

                    for j in range(int(config['FRAMES_PER_CLIP'])):
                        orig = original[k, :, j, :, :].cpu().permute(
                            1, 2, 0).numpy()
                        comp = inputs[k, :, j, :, :].cpu().permute(
                            1, 2, 0).numpy()
                        psnr_score = calculate_psnr(orig, comp)
                        psnr_score_list.append(psnr_score)
                        ssim_score = calculate_ssim(orig, comp)
                        ssim_score_list.append(ssim_score)
                        mae_score = calculate_mae(orig, comp)
                        mae_score_list.append(mae_score)
                        rmse_score = calculate_rmse(orig, comp)
                        rmse_score_list.append(rmse_score)

                    ave_psnr = sum(psnr_score_list) / len(psnr_score_list)
                    ave_ssim = sum(ssim_score_list) / len(ssim_score_list)
                    ave_mae = sum(mae_score_list) / len(mae_score_list)
                    ave_rmse = sum(rmse_score_list) / len(rmse_score_list)

                    experiment.log_metric('psnr', ave_psnr, step=i)
                    experiment.log_metric('ssim', ave_ssim, step=i)
                    experiment.log_metric('mae', ave_mae, step=i)
                    experiment.log_metric('rmse', ave_rmse, step=i)
                    i += 1


def main():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="VaG6pF4qhcqKJOux0daNkIz2C",
        project_name="jpeg-compression",
        workspace="ohtani",
    )

    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config['kinetics']

    parser = argparse.ArgumentParser()
    parser.add_argument('--compression_rate', type=float,
                        default=100.0, help='Please Enter a number')
    parser.add_argument('--path', action='store',
                        default=config['DATA_PATH'],
                        help='file path it is written urls')
    parser.add_argument('--use_compression', action='store_true', help='')
    args = parser.parse_args()
    # simple_checkの引数にargparseを入れる
    compression = args.compression_rate
    print('jpeg_compression:', compression)

    root_path = args.path
    if os.path.exists(root_path):
        print('PATH:', root_path)
    else:
        print('No File')

    en = args.use_compression
    print('en:', en)

    params = get_params(config)
    experiment.log_parameters(params)
    if(en):
        experiment.log_parameter('compression', compression)
    else:
        experiment.log_parameter('compression', -10)
    experiment.log_parameter('path_root', root_path)

    experiment.add_tag('score')

    sample_check(compression, root_path, config, en, experiment)


if __name__ == '__main__':
    main()
