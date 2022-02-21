from comet_ml import Experiment
from ast import parse
from logging import NullHandler, root
import re
from imgaug.augmenters import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
import numpy as np

from torchvision import transforms

from pytorchvideo.models import x3d
from pytorchvideo.data import RandomClipSampler, UniformClipSampler
from pytorchvideo.data import Ucf101, RandomClipSampler, UniformClipSampler, Kinetics
# from segmentation_dataset import segUcf101, segkinetics

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
from torchvision.transforms.functional import scale

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

from compression_transform import SameClipSampler

# from compression_transform import (
#     JpegCompression,
#     CompCompose,
#     BothCompose,
#     CompApplyTransformToKey,
#     CompUniformTemporalSubsample,
#     CompLambda,
#     BothLambda,
#     CompNormalize,
#     CompRandomShortSideScale,
#     CompRandomCrop,
#     CompRandomHorizontalFlip,
#     CompRemoveKey,
#     CompShortSideScale,
#     CompCenterCrop,
#     SameClipSampler
# )


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


class JpegCompression():
    def __init__(self, compression_strength, size_params):
        self.compression_strength = compression_strength
        self.size_params = size_params

    def __call__(self, img):
        # args = Args()
        aug = iaa.JpegCompression(self.compression_strength)
        for i in range(int(self.size_params['NUM_FRAMES'])):
            image = img[:, i, :, :].permute(1, 2, 0)
            # xx = xx.to('cpu').detach().numpy().copy()
            image = image.numpy()
            # xx = xx * 255
            # xx = xx.numpy()
            image = image.clip(0, 255).astype(np.uint8)
            # xx = xx.astype(np.uint8)
            tmp = aug.augment_image(image)
            tmp = np.asarray(tmp).astype('float32')
            tmp = tmp / 255
            img[:, i, :, :] = torch.from_numpy(tmp).permute(2, 0, 1)
            # img[:,i,:,:] = torch.from_numpy(tmp).clone().permute(2,0,1)
        return img


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    https://pytorch.org/hub/facebookresearch_pytorchvideo_slowfast/
    """

    def __init__(self):
        super().__init__()
        self.slowfast_alpha = 4

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def use_compression(args, size_params):
    compression_strength = args.compression_strength
    en = args.use_compression
    jpeg_compression = None
    if (en):
        print('圧縮がかかっています')
        jpeg_compression = JpegCompression(
            compression_strength, size_params)
    else:
        print('圧縮はかかっていません')
        jpeg_compression = transforms.Lambda(lambda x: x / 255.)
    return jpeg_compression


def use_packpathway(args):
    model_name = args.model_name
    if "slowfast" in model_name:
        packpathway = PackPathway()
    else:
        packpathway = transforms.Lambda(lambda x: x)
    return packpathway


def get_kinetics(args, jpeg_compression, size_params, packpathway):
    """
    Kinetics400のデータセットを取得
    Args:
        subset (str): "train" or "val"
    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """

    subset = args.subset
    root_path = args.path
    transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(
                    int(size_params['NUM_FRAMES'])),
                # print(config['NUM_WORKERS']),
                # TODO: 受け渡されたjpegcompressionオブジェクトをここに入れる
                jpeg_compression,
                # transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                # RandomShortSideScale(min_size=256, max_size=320,),
                ShortSideScale(size=int(size_params['SIDE_SIZE'])),
                CenterCrop(int(size_params['CROP_SIZE'])),
                # RandomHorizontalFlip(),
                # PackPathway(),
                packpathway,
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x),
        ),
        RemoveKey("audio"),
    ])

    # root_kinetics = '/mnt/NAS-TVS872XT/dataset/Kinetics400/'
    root_kinetics = root_path
    # The duration of the input clip is also specific to the model.
    num_frames = int(size_params['NUM_FRAMES'])
    sampling_rate = int(size_params['SAMPLIMG_RATE'])
    frames_per_second = 30
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    dataset = Kinetics(
        data_path=root_kinetics + subset,
        video_path_prefix=root_kinetics + subset,
        clip_sampler=SameClipSampler(
            clip_duration=float(clip_duration)),
        video_sampler=SequentialSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def get_ucf101(args, jpeg_compression, size_params, packpathway):
    """
    ucf101のデータセットを取得
    Args:
        subset (str): "train" or "val"
    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """

    subset = args.subset
    root_path = args.path
    subset_root_ucf101 = 'ucfTrainTestlist/trainlist01.txt'
    if subset == "test":
        subset_root_ucf101 = 'ucfTrainTestlist/testlist.txt'

    # args = Args()
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(int(size_params['NUM_FRAMES'])),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320,),
                RandomCrop(224),
                RandomHorizontalFlip(),
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x - 1),
        ),
        RemoveKey("audio"),
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(
                    int(size_params['NUM_FRAMES'])),
                # print(config['NUM_WORKERS']),
                # TODO: 受け渡されたjpegcompressionオブジェクトをここに入れる
                jpeg_compression,
                # transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                # RandomShortSideScale(min_size=256, max_size=320,),
                ShortSideScale(size=int(size_params['SIDE_SIZE'])),
                CenterCrop(int(size_params['CROP_SIZE'])),
                # RandomHorizontalFlip(),
                # PackPathway(),
                packpathway,
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x - 1),
        ),
        RemoveKey("audio"),
    ])

    transform = train_transform if subset == "train" else val_transform

    root_ucf101 = root_path
    # root_ucf101 = '/mnt/NAS-TVS872XT/dataset/UCF101/'

    num_frames = int(size_params['NUM_FRAMES'])
    sampling_rate = int(size_params['SAMPLIMG_RATE'])
    frames_per_second = 30
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    dataset = Ucf101(
        data_path=root_ucf101 + subset_root_ucf101,
        video_path_prefix=root_ucf101 + 'video/',
        clip_sampler=SameClipSampler(
            clip_duration=float(clip_duration)),
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


def get_dataset(args, jpeg_compression, size_params, pathpackway):
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
    dataset = args.dataset
    print('dataset:', dataset)
    # subset = args.subset
    if dataset == "Kinetics400":
        return get_kinetics(args, jpeg_compression, size_params, pathpackway)
    elif dataset == "UCF101":
        return get_ucf101(args, jpeg_compression, size_params, pathpackway)
    else:
        print('dataloader error')


def get_model(args, pretrained):
    """
    pytorchvideoからモデルを取得
    Args:
        model (str): "x3d_m"(UCF101用) or "slow_r50"(Kinetics400用)
        pretrained (bool): "True" or "False"
    Returns:
        model: 取得したモデル
    """
    model_name = args.model_name
    dataset = args.dataset
    model = torch.hub.load(
        'facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
    # do_fine_tune = True
    # if do_fine_tune:
    #     for param in model.parameters():
    #         param.requires_grad = False
    if dataset == 'UCF101':
        model.blocks[5].proj = nn.Linear(
            model.blocks[5].proj.in_features, 101)
    # if model_name == "x3d_m":
    #     model.blocks[5].proj = nn.Linear(
    #         model.blocks[5].proj.in_features, 400)
    return model


def dataset_check(dataset, subset):
    """
    取得したローダーの挙動を確認
    Args:
        dataset (str): "Kinetics400" or "UCF101"
        subset (str): "train" or "val"
    """
    dataset = get_dataset("Kinetics400", "val")
    loader = make_loader(dataset)
    print("len:{}".format(len(loader)))
    for i, batch in enumerate(loader):
        if i == 0:
            print(batch.keys())
            print(batch['video'].shape)
        print(batch['label'].cpu().numpy())
        if i > 4:
            break

# comet debug用


def get_params(config, size_params):
    params = {
        "batch_size": int(config['BATCH_SIZE']),
        "num_workers": int(config['NUM_WORKERS']),
        "size_size": int(size_params['SIDE_SIZE']),
        "crop_size": int(size_params['CROP_SIZE']),
        "num_frames": int(size_params['NUM_FRAMES']),
        "sampling_rate": int(size_params['SAMPLIMG_RATE'])
        # "model": config['MODEL']
    }

    return params


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L363-L380
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) == torch.Tensor:
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def sample_check(args, config, experiment, size_params):
    """学習済みモデルにサンプルデータ100個を流し込んで挙動を確認"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = get_model(args, True)
    model = model.to(device)
    # TODO: jpegcompressionオブジェクトここで作ってget_datasetに渡す
    # jpeg_compression = JpegCompression
    # compression = jpeg_compression()
    # print("sample_check:", compression)
    jpeg_compression = use_compression(args, size_params)
    pathpackway = use_packpathway(args)
    # print('type(jpeg_compression):', type(jpeg_compression))
    dataset = get_dataset(args, jpeg_compression, size_params, pathpackway)

    # dataset.video_sampler._num_samples = 100
    sample_loader = make_loader(dataset, config)

    acc1_list = []
    acc5_list = []
    log_top1 = AverageMeter()
    log_top5 = AverageMeter()
    video_num = 0
    model.eval()
    with experiment.validate(), torch.no_grad():
        with tqdm(enumerate(sample_loader),
                  total=len(sample_loader),
                  leave=True) as pbar_batch:

            for batch_idx, batch in pbar_batch:
                # print(batch['video_name'])
                # print(batch['video'].shape)
                # print('batch_idx:', batch_idx)
                # print(1)
                # sub_inputs = PackPathway(batch["video"])
                # inputs = [b.to(device) for b in sub_inputs]
                if "slowfast" in args.model_name:
                    inputs = [b.to(device) for b in batch['video']]
                    experiment.log_parameter(
                        'num_frame_0', inputs[0][0, 0, :, :, :].shape)
                    experiment.log_parameter(
                        'num_frame_1', inputs[1][0, 0, :, :, :].shape)
                    current_batch_size = inputs[0].size()[0]
                else:
                    inputs = batch["video"].to(device)
                    experiment.log_parameter(
                        'frame_size_0', inputs[0, 0, 0, :, :].shape)
                    # print('shape:', inputs[:, 0, 0, 0, 0].shape)
                    current_batch_size = inputs.size()[0]
                    # print('inputs.shape:', inputs.size()[0])
                    # print(inputs[:, 0, 0, 0, 0].shape)

                # inputs = [b.to(device) for b in batch['video']]
                labels = batch["label"].to(device)
                outputs = model(inputs)
                # print(inputs[0, 0, 0, :, :].shape)
                # print(inputs.shape[4])
                # preds = torch.squeeze(outputs.max(dim=1)[1])
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                log_top1.update(acc1, current_batch_size)
                log_top5.update(acc5, current_batch_size)

                acc1_list.append(log_top1.val)
                acc5_list.append(log_top5.val)
                pbar_batch.set_postfix_str(' | acc top1={:6.04f} top5={:6.04f}'''
                                           .format(log_top1.val, log_top5.val))
                video_num += current_batch_size
                experiment.log_parameter('batch_idx', batch_idx)

        # accuracy1 = sum(acc1_list) / len(acc1_list)
        # accuracy5 = sum(acc5_list) / len(acc5_list)
        # print("sum_acc1:{}, sum_acc5:{}".format(
        #     sum(acc1_list), sum(acc5_list)))
        # print("len_acc1:{}, len_acc5:{}".format(
        #     len(acc1_list), len(acc5_list)))
        print("acc1:{}, acc5:{}".format(log_top1.avg, log_top5.avg))
        experiment.log_metric("acc1", log_top1.avg)
        experiment.log_metric("acc5", log_top5.avg)
        experiment.log_parameter('video_num', video_num)


def main():
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="VaG6pF4qhcqKJOux0daNkIz2C",
        project_name="jpeg-compression",
        workspace="ohtani",
    )

    # experiment = Experiment(
    #     api_key="VaG6pF4qhcqKJOux0daNkIz2C",
    #     project_name="ffmpeg",
    #     workspace="ohtani",
    # )

    # print(config['CLIP_DURATION'])
    # print(type(config['CLIP_DURATION']))
    # TODO: argparseをここに入れる
    # parser = argparse.ArgumentParser(description='jpegcompressionの圧縮の強さ')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store',
                        default='Kinetics400', help='')
    parser.add_argument('-sb', '--subset',
                        action='store', default='val', help='')
    parser.add_argument('--path', action='store',
                        # default=config['DATA_PATH'],
                        help='file path it is written urls')
    parser.add_argument('--use_compression', action='store_true', help='')
    parser.add_argument('-cs', '--compression_strength', type=float,
                        default=99.0, help='Please Enter a number')
    parser.add_argument('--model_name', action='store',
                        default='x3d_m', help='Enter model name')
    parser.add_argument('--q_use', action='store_true', help=' ')
    args = parser.parse_args()

    # simple_checkの引数にargparseを入れる
    compression_strength = args.compression_strength
    print('compression_strength:', compression_strength)

    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config[args.dataset]
    # parser.add_argument('--path', action='store',
    #                     default=config['DATA_PATH'],
    #                     help='file path it is written urls')
    # args = parser.parse_args()

    root_path = args.path
    if root_path is None:
        root_path = config['DATA_PATH']
        args.path = config['DATA_PATH']
    # root_path = config['DATA_PATH']
    if os.path.exists(root_path):
        print('PATH:', root_path)
    else:
        print('No File', root_path)

    en = args.use_compression
    print('en:', en)

    model_name = args.model_name
    experiment.log_parameter('model', model_name)
    print('model_name:', model_name)

    size_params = configparser.ConfigParser()
    size_params.read('model.ini')
    size_params = size_params[model_name]

    params = get_params(config, size_params)
    experiment.log_parameters(params)
    if(en):
        experiment.log_parameter('compression', compression_strength)
    else:
        experiment.log_parameter('compression', -10)
    experiment.log_parameter('path_root', root_path)

    experiment.add_tag('top1,5_ver.11')

    if args.q_use:
        q = os.path.split(os.path.split(root_path)[0])[1]
        experiment.log_parameter('q', q)
        print('q:', q)
    else:
        experiment.log_parameter('q', 0)
        print('q:', 0)

    sample_check(args, config, experiment, size_params)


if __name__ == '__main__':
    main()
