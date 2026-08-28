import torch.nn as nn

import segmentation_models_pytorch as smp

class UNet2D(nn.Module):
    """
    A 2D UNet model with various architectures built by segmentation_models_pytorch.
    """

    def __init__(self,
                 num_channels=1, num_classes=2,
                 architecture='Unet',
                 encoder_name='resnet34',
                 pretrained=True):
        super().__init__()

        self.num_classes = num_classes
        self.architecture = architecture
        self.encoder_name = encoder_name

        encoder_weights = 'imagenet' if pretrained else None

        architectures = {
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
        model_builder = architectures.get(architecture)
        if model_builder is None:
            raise ValueError(f"Unknown architecture '{architecture}'. Expected one of: {sorted(architectures)}")

        self.model = model_builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=num_channels,
            classes=num_classes,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(x))
