import torch
from collections import OrderedDict
import torch.nn as nn


# class X3D(nn.Module):
#     def __init__(self, num_classes,  pretrain="scratch", model_name='x3d_m'):
#         super().__init__()
#         self.pretrained = False
#         self.model_name = model_name
#         if pretrain == "pretrain":
#             self.pretrained = True
#         model = torch.hub.load('facebookresearch/pytorchvideo',
#                                self.model_name, pretrained=self.pretrained)
#         for param in model.parameters():
#             param.requires_grad = True
#         self.net_bottom = nn.Sequential(
#             model.blocks[0],
#             model.blocks[1],
#             model.blocks[2],
#             model.blocks[3],
#             model.blocks[4]
#         )
#         self.net_top = nn.Sequential(
#             model.blocks[5].pool,
#             model.blocks[5].dropout
#         )
#         self.linear = nn.Linear(2048, num_classes)

#     def forward(self, x: torch.Tensor):
#         x = self.net_bottom(x)
#         x = self.net_top(x)
#         x = x.permute(0, 2, 3, 4, 1)
#         x = self.linear(x)
#         x = x.view(x.size(0), -1)
#         return x

def get_model_path(args, model_params):
    dataset = args.dataset
    # model_name = args.model_name
    if dataset == 'UCF101':
        model_path = model_params['MODEL_PATH']
    checkpoint = torch.load(model_path)

    state_dict = checkpoint
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # これで先頭のmodule.を消す
        new_state_dict[name] = v

    return checkpoint
