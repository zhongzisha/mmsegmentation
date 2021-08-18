# Copyright (c) OpenMMLab. All rights reserved.
from .cgnet import CGNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .swin import SwinTransformer
from .unet import UNet
from .vit import VisionTransformer

__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer'
]

try:
    from .vit_in_SETR import VisionTransformerInSETR
    from .vit_mla import VIT_MLA
    __all__ += ['VisionTransformerInSETR', 'VIT_MLA']
except ImportError:
    pass
