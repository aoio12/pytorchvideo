import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv3d
# import torchinfo


class ResNet_R50(nn.Module):
    def __init__(self, num_classes,  pretrain=False, model_name='slow_r50'):
        super().__init__()
        self.pretrained = pretrain
        self.model_name = model_name
        # if pretrain == "pretrain":
        #     self.pretrained = True
        model = torch.hub.load('facebookresearch/pytorchvideo',
                               self.model_name, pretrained=self.pretrained)
        for param in model.parameters():
            param.requires_grad = True
        self.net_bottom = nn.Sequential(
            model.blocks[0],
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
            model.blocks[4]
        )
        self.net_top = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )
        self.linear = model.blocks[5].proj
        # self.linear = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.net_bottom(x)
        x = self.net_top(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.view(x.size(0), -1)
        return x

# B, C, F, H, W -> B, F, H, W, C
