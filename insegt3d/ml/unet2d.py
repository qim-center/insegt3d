import os
import torch
import torch.nn as nn
from pathlib import Path

import segmentation_models_pytorch as smp

ARCHITECTURES = {
    'Unet': smp.Unet,
    'U-Net++': smp.UnetPlusPlus,
    'FPN': smp.FPN,
    'PSPNet': smp.PSPNet,
    'DeepLabV3': smp.DeepLabV3,
    'DeepLabV3+': smp.DeepLabV3Plus,
    'Linknet': smp.Linknet,
    'MAnet': smp.MAnet,
    'PAN': smp.PAN,
    'UPerNet': smp.UPerNet,
    'Segformer': smp.Segformer,
    'DPT': smp.DPT,
}

BATCHNORM_MOMENTUM = 0.05

def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)

def save_checkpoint(model, level, path):
    tmp = Path(path).with_suffix('.tmp')
    torch.save({'config': model.config, 'level': level, 'state_dict': model.state_dict()}, tmp)
    os.replace(tmp, path)

class UNet2D(nn.Module):
    """2D segmentation model built from any segmentation_models_pytorch architecture."""

    def __init__(self,
                 num_channels=1, num_classes=2,
                 architecture='Unet',
                 encoder_name='resnet34',
                 pretrained=True):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.architecture = architecture
        self.encoder_name = encoder_name

        encoder_weights = 'imagenet' if pretrained else None

        self.model = ARCHITECTURES[architecture](
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=num_channels,
            classes=num_classes,
        )

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = BATCHNORM_MOMENTUM

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(**checkpoint['config'], pretrained=False)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    @property
    def config(self):
        return {
            'num_channels': self.num_channels,
            'num_classes': self.num_classes,
            'architecture': self.architecture,
            'encoder_name': self.encoder_name,
        }

    def set_classes(self, keep, added=0):
        conv = [m for m in self.model.segmentation_head.modules() if isinstance(m, nn.Conv2d)][-1]
        weight, bias = conv.weight.data[keep], conv.bias.data[keep]
        # New classes start with zero weights and a low bias, so existing predictions are unchanged
        conv.weight = nn.Parameter(torch.cat([weight, weight.new_zeros((added, *weight.shape[1:]))]))
        conv.bias = nn.Parameter(torch.cat([bias, bias.new_full((added,), bias.min().item() - 2)]))
        conv.out_channels = self.num_classes = len(keep) + added

    def logits(self, x):
        return self.model(x)

    def forward(self, x):
        return self.logits(x).softmax(1)
