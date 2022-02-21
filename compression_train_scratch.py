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
from torchvision import models

from pytorchvideo.models import x3d
from pytorchvideo.data import RandomClipSampler, UniformClipSampler
from pytorchvideo.data import Ucf101, RandomClipSampler, UniformClipSampler, Kinetics

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
from typing import Type
import itertools
import os
import os.path as osp
import pickle

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import argparse
import configparser
from fractions import Fraction

from compression_transform import SameClipSampler
from dataset_model import get_model_path
from kinetics_train.x3d_m import X3D_ABN
from kinetics_train.resnet import ResNet_R50
from kinetics_train.slowfast_r101 import Slowfast_R101
from ucf_train import x3d_m, resnet, slowfast_r101
from ucf_train.model_save import save_checkpoint


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
    def __init__(self, compression_strength, model_params):
        self.compression_strength = compression_strength
        self.model_params = model_params

    def __call__(self, img):
        # args = Args()
        aug = iaa.JpegCompression(self.compression_strength)
        for i in range(int(self.model_params['NUM_FRAMES'])):
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


def use_compression(args, model_params):
    compression_strength = args.compression_strength
    en = args.use_compression
    jpeg_compression = None
    if (en):
        print('JPEG圧縮がかかっています')
        jpeg_compression = JpegCompression(
            compression_strength, model_params)
    else:
        print('JPEG圧縮はかかっていません')
        jpeg_compression = transforms.Lambda(lambda x: x / 255.)
    return jpeg_compression


def use_packpathway(args):
    model_name = args.model_name
    if "slowfast" in model_name:
        packpathway = PackPathway()
    else:
        packpathway = transforms.Lambda(lambda x: x)
    return packpathway


def get_kinetics(subset, args, jpeg_compression, model_params, packpathway):
    """
    Kinetics400のデータセットを取得
    Args:
        subset (str): "train" or "val"
    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """

    # subset = args.subset
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(int(model_params['NUM_FRAMES'])),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320,),
                RandomCrop(224),
                RandomHorizontalFlip(),
                packpathway,
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x),
        ),
        RemoveKey("audio"),
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(
                    int(model_params['NUM_FRAMES'])),
                # TODO: 受け渡されたjpegcompressionオブジェクトをここに入れる
                jpeg_compression,
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(size=int(model_params['SIDE_SIZE'])),
                CenterCrop(int(model_params['CROP_SIZE'])),
                packpathway,
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x),
        ),
        RemoveKey("audio"),
    ])

    # The duration of the input clip is also specific to the model.
    num_frames = int(model_params['NUM_FRAMES'])
    sampling_rate = int(model_params['SAMPLIMG_RATE'])
    frames_per_second = 30
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    root_kinetics = train_data_path if subset == "train" else val_data_path
    transform = train_transform if subset == "train" else val_transform
    # clipsampler = RandomClipSampler if subset == 'train' else SameClipSampler
    # sampler_video = RandomSampler if subset == "train" else SequentialSampler

    dataset = Kinetics(
        data_path=root_kinetics + subset,
        video_path_prefix=root_kinetics + subset,
        clip_sampler=RandomClipSampler(
            clip_duration=float(clip_duration)),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def get_ucf101(args, jpeg_compression, model_params, packpathway):
    """
    ucf101のデータセットを取得
    Args:
        subset (str): "train" or "test"
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
                UniformTemporalSubsample(int(model_params['NUM_FRAMES'])),
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

    test_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(
                    int(model_params['NUM_FRAMES'])),
                # TODO: 受け渡されたjpegcompressionオブジェクトをここに入れる
                jpeg_compression,
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(size=int(model_params['SIDE_SIZE'])),
                CenterCrop(int(model_params['CROP_SIZE'])),
                packpathway,
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x - 1),
        ),
        RemoveKey("audio"),
    ])

    transform = train_transform if subset == "train" else test_transform
    clipsampler = RandomClipSampler if subset == 'train' else SameClipSampler

    root_ucf101 = root_path
    # root_ucf101 = '/mnt/NAS-TVS872XT/dataset/UCF101/'

    num_frames = int(model_params['NUM_FRAMES'])
    sampling_rate = int(model_params['SAMPLIMG_RATE'])
    frames_per_second = 30
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    dataset = Ucf101(
        data_path=root_ucf101 + subset_root_ucf101,
        video_path_prefix=root_ucf101 + 'video/',
        clip_sampler=clipsampler(
            clip_duration=float(clip_duration)),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )
    return dataset


def make_loader(dataset, config, args):
    """
    データローダーを作成
    Args:
        dataset (pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset): get_datasetメソッドで取得したdataset
    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    # args = Args()
    loader = DataLoader(LimitDataset(dataset),
                        batch_size=args.batch_size,
                        drop_last=True,
                        num_workers=int(config['NUM_WORKERS']))
    return loader


def get_dataset(args, jpeg_compression, model_params, packpathway):
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
        return get_kinetics(args, jpeg_compression, model_params, packpathway)
    elif dataset == "UCF101":
        return get_ucf101(args, jpeg_compression, model_params, packpathway)
    else:
        print('dataloader error')


def get_model(args, config, pretrained):
    """
    pytorchvideoからモデルを取得
    Args:
        model (str): "x3d_m"(UCF101用) or "slow_r50"(Kinetics400用)
        pretrained (bool): "True" or "False"
    Returns:
        model: 取得したモデル
    """

    num_classes = int(config['NUM_CLASSES'])
    if args.dataset == 'Kinetics400':
        if "slowfast" in args.model_name:
            model = Slowfast_R101(num_classes, pretrain=pretrained,
                                  model_name=args.model_name)
        elif "x3d" in args.model_name:
            model = X3D_ABN(num_classes, pretrain=pretrained,
                            model_name=args.model_name)
        else:
            model = ResNet_R50(num_classes, pretrain=pretrained,
                               model_name=args.model_name)
    else:
        if "slowfast" in args.model_name:
            model = slowfast_r101.Slowfast_R101(
                num_classes, pretrain=pretrained, model_name=args.model_name)
        elif "x3d" in args.model_name:
            model = x3d_m.X3D_ABN(num_classes, pretrain=pretrained,
                                  model_name=args.model_name)
        else:
            model = resnet.ResNet_R50(num_classes, pretrain=pretrained,
                                      model_name=args.model_name)
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


def get_params(config, model_params):
    params = {
        # "batch_size": int(config['BATCH_SIZE']),
        # "batch_size": 8,
        "num_workers": int(config['NUM_WORKERS']),
        "size_size": int(model_params['SIDE_SIZE']),
        "crop_size": int(model_params['CROP_SIZE']),
        "num_frames": int(model_params['NUM_FRAMES']),
        "sampling_rate": int(model_params['SAMPLIMG_RATE'])
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


def train(args, config, experiment, model_params):
    """学習済みモデルにサンプルデータ100個を流し込んで挙動を確認"""
    gpu = args.gpu
    device = torch.device(
        "cuda:" + gpu if torch.cuda.is_available() else "cpu")
    jpeg_compression = use_compression(args, model_params)
    packpathway = use_packpathway(args)
    train_dataset = get_kinetics(
        "train", args, jpeg_compression, model_params, packpathway)
    val_dataset = get_kinetics(
        "val", args, jpeg_compression, model_params, packpathway)

    train_loader = make_loader(train_dataset, config, args)
    val_loader = make_loader(val_dataset, config, args)

    model = get_model(args, config, False)

    model = model.to(device)
    torch.backends.cudnn.benchmark = True
    lr = 0.0001
    experiment.log_parameter('lr', lr)

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=5e-5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=5e-5)
    criterion = nn.CrossEntropyLoss()
    num_epochs = args.epoch

    step = 0
    best_acc = 0

    with tqdm(range(num_epochs)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description("[Epoch %d]" % (epoch))

            """Training mode"""

            train_loss = AverageMeter()
            train_top1 = AverageMeter()
            train_top5 = AverageMeter()

            with tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      leave=True) as pbar_train_batch:

                model.train()

                for batch_idx, train_batch in pbar_train_batch:
                    pbar_train_batch.set_description(
                        "[Epoch :{}]".format(epoch))

                    if "slowfast" in args.model_name:
                        inputs = [b.to(device) for b in train_batch['video']]
                        bs = inputs[0].size(0)
                    else:
                        inputs = train_batch['video'].to(device)
                        bs = inputs.size(0)
                    labels = train_batch['label'].to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss.update(loss, bs)
                    train_acc1, train_acc5 = accuracy(
                        outputs, labels, topk=(1, 5))
                    train_top1.update(train_acc1, bs)
                    train_top5.update(train_acc5, bs)

                    pbar_train_batch.set_postfix_str(
                        ' | batch_loss={:6.04f} , batch_top1={:6.04f}'
                        ' | loss_avg={:6.04f} , top1_avg={:6.04f}, top5_avg={:6.04f}'
                        ''.format(
                            train_loss.val, train_top1.val,
                            train_loss.avg, train_top1.avg, train_top5.avg
                        ))

                    experiment.log_metric(
                        "batch_top1accuracy", train_top1.val, step=step)
                    experiment.log_metric(
                        "batch_top5accuracy", train_top5.val, step=step)
                    step += 1

            """Val mode"""

            val_loss_list = AverageMeter()
            val_top1 = AverageMeter()
            val_top5 = AverageMeter()

            model.eval()
            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_loader):
                    if "slowfast" in args.model_name:
                        inputs = [b.to(device) for b in val_batch['video']]
                        bs = inputs[0].size(0)
                    else:
                        inputs = val_batch['video'].to(device)
                        bs = inputs.size(0)
                    labels = val_batch['label'].to(device)

                    val_outputs = model(inputs)
                    val_loss = criterion(val_outputs, labels)

                    val_loss_list.update(val_loss, bs)
                    val_acc1, val_acc5 = accuracy(
                        val_outputs, labels, topk=(1, 5))
                    val_top1.update(val_acc1, bs)
                    val_top5.update(val_acc5, bs)
            """Finish Val mode"""

            """save model"""
            if best_acc < val_top1.avg:
                best_acc = val_top1.avg
                is_best = True
            else:
                is_best = False

            dir_path = os.path.join(
                "model_path/scratch", args.dataset, args.model_name)
            save_checkpoint(model, is_best, filename=args.model_pth + ".pth",
                            best_model_file=args.model_pth + "_best.pth", dir_data_name=dir_path)
            # save_checkpoint(model, is_best, filename="original.pth",
            #                 best_model_file="original_best.pth", dir_data_name="model_path/Kinetics400/x3d_m")

            pbar_epoch.set_postfix_str(
                ' | train_loss={:6.04f}, train_top1_avg={:6.04f}'
                ' | val_loss={:6.04f}, val_top1_val={:6.04f}, val_top1_avg={:6.04f}, val_top5_avg={:6.04f}'
                ''.format(
                    train_loss.avg,
                    train_top1.avg,
                    val_loss_list.avg,
                    val_top1.val,
                    val_top1.avg,
                    val_top5.avg)
            )

            """train log"""
            experiment.log_metric("epoch_train_top1",
                                  train_top1.avg,
                                  step=epoch + 1)
            experiment.log_metric("epoch_train_top5",
                                  train_top5.avg,
                                  step=epoch + 1)
            experiment.log_metric("epoch_train_loss",
                                  train_loss.avg,
                                  step=epoch + 1)
            """val los"""
            experiment.log_metric("val_top1",
                                  val_top1.avg,
                                  step=epoch + 1)
            experiment.log_metric("val_top5",
                                  val_top5.avg,
                                  step=epoch + 1)
            experiment.log_metric("val_loss",
                                  val_loss_list.avg,
                                  step=epoch + 1)


def main():
    # Create an experiment with your api key
    # experiment = Experiment(
    #     api_key="VaG6pF4qhcqKJOux0daNkIz2C",
    #     project_name="jpeg-compression",
    #     workspace="ohtani",
    # )

    experiment = Experiment(
        api_key="VaG6pF4qhcqKJOux0daNkIz2C",
        project_name="ffmpeg",
        workspace="ohtani",
    )

    # parser = argparse.ArgumentParser(description='jpegcompressionの圧縮の強さ')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='store',
                        default='Kinetics400', help='')
    # parser.add_argument('-sb', '--subset',
    #                     action='store', default='val', help='')
    parser.add_argument('--path', action='store',
                        default=None,
                        help='file path it is written urls')
    parser.add_argument('--train_data_path',
                        action='store', default=None, help='')
    parser.add_argument('--train_crf', type=int, default=-10, help="")
    parser.add_argument('--train_gop', type=int, default=-10, help="")
    parser.add_argument('--val_data_path', action='store',
                        default=None, help='')
    parser.add_argument('--val_crf', type=int, default=-10, help="")
    parser.add_argument('--val_gop', type=int, default=-10, help="")
    parser.add_argument('--use_compression', action='store_true', help='')
    parser.add_argument('-cs', '--compression_strength', type=float,
                        default=99.0, help='Please Enter a number')
    parser.add_argument('--model_name', action='store',
                        default='x3d_m', help='Enter model name')
    # parser.add_argument('--q_use', action='store_true', help=' ')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='number of gpus used for conversion.')
    parser.add_argument('--crf', type=int, default=-10, help="")
    parser.add_argument('--gop', type=int, default=-10, help="")
    parser.add_argument('--model_pth', action='store',
                        default='original', help='Enter save model path')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()

    # simple_checkの引数にargparseを入れる

    config = configparser.ConfigParser()
    config.read('config.ini')
    config = config[args.dataset]

    # trainデータパスチェック
    if args.train_data_path is None:
        args.train_data_path = config['DATA_PATH']
    if os.path.exists(args.train_data_path):
        print('train_data_path:', args.train_data_path)
    else:
        print('No File', args.train_data_path)

    # valデータパスチェック
    if args.val_data_path is None:
        args.val_data_path = config['DATA_PATH']
    if os.path.exists(args.val_data_path):
        print('val_data_path:', args.val_data_path)
    else:
        print('No File', args.val_data_path)

    if not (args.use_compression):
        args.compression_strength = -10
    if args.batch_size == 1:
        args.batch_size = int(config['BATCH_SIZE'])
    if args.epoch == 1:
        args.epoch = int(config['NUM_EPOCH'])

    print('dataset_name:', args.dataset)
    # print('subset:', args.subset)
    print('use_compression:', args.use_compression)
    print('compression_strength:', args.compression_strength)
    print('model_name:', args.model_name)

    dir_path = os.path.join(
        "model_path/scratch", args.dataset, args.model_name)
    model_save_path = osp.join(dir_path, args.model_pth + ".pth",)
    print('save_path:', model_save_path)
    model_params = configparser.ConfigParser()
    model_params.read('model.ini')
    model_params = model_params[args.model_name]

    params = get_params(config, model_params)
    experiment.log_parameters(params)

    parameters = {
        "dataset_name": args.dataset,
        "train_data_path": args.train_data_path,
        "val_data_path": args.val_data_path,
        "model": args.model_name,
        "train_crf": args.train_crf,
        "train_gop": args.train_gop,
        "val_crf": args.val_crf,
        "val_gop": args.val_gop,
        "compression_strength": args.compression_strength,
        "use_compression": args.use_compression,
        "model_pth": args.model_pth,
        "batch_size": args.batch_size,
        "epochs": args.epoch,
    }
    experiment.log_parameters(parameters)
    experiment.add_tag('kinetics_train_scratch')

    train(args, config, experiment, model_params)


if __name__ == '__main__':
    main()
