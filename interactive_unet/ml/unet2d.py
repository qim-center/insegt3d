import torch.nn as nn

import segmentation_models_pytorch as smp

class UNet2D(nn.Module):
    """
    A 2D UNet model with various architectures built by segmentation_models_pytorch.
    """
    
    def __init__(self,
                 num_channels=1, num_classes=2,
                 architecture='U-Net',
                 encoder_name='resnet34',
                 pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes

        encoder_weights = 'imagenet' if pretrained else None

        model_builder = {
            'U-Net': smp.Unet,
            'U-Net++': smp.UnetPlusPlus,
            'FPN': smp.FPN,
            'PSPNet': smp.PSPNet,
            'DeepLabV3': smp.DeepLabV3,
            'DeepLabV3+': smp.DeepLabV3Plus,
            'LinkNet': smp.Linknet,
            'MA-Net': smp.MAnet,
            'PAN': smp.PAN,
            'UPerNet': smp.UPerNet,
            'Segformer': smp.Segformer,
        }[architecture]

        self.model = model_builder(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=num_channels,
            classes=num_classes,
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        output = self.softmax(self.model(x))
        
        return output

    def set_num_classes(self, num_classes):
        """
        Update the number of output classes, changing only 
        the head and preserving the model weights.
        """

        # Skip if same
        if num_classes == self.num_classes:
            return

        # Get old segmentation head
        head = self.model.segmentation_head
        layers = list(head.children())
        last = layers[-1]

        # Build new segmentation head
        new_conv = nn.Conv2d(
            in_channels=last.in_channels,
            out_channels=num_classes,
            kernel_size=last.kernel_size,
            stride=last.stride,
            padding=last.padding,
            bias=(last.bias is not None),
        )

        # Initialize new weights
        nn.init.xavier_uniform_(new_conv.weight)
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

        # Preserve existing classes weights
        old_classes = min(self.num_classes, num_classes)
        new_conv.weight.data[:old_classes] = last.weight.data[:old_classes]
        if last.bias is not None:
            new_conv.bias.data[:old_classes] = last.bias.data[:old_classes]

        # Update segmentation head with new one
        layers[-1] = new_conv
        self.model.segmentation_head = nn.Sequential(*layers)

        # Update num_classes
        self.num_classes = num_classes