import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..util.unet import UNet


class UNetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super(UNetWrapper, self).__init__()
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d}:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)

        return fn_output


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super(SegmentationAugmentation, self).__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def _build2dTransformMatrix(self):
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random * 2 - 1)
                transform_t[i, i] = 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

            transform_t @= rotation_t

        return transform_t

    def forward(self, input_g, label_g):
        transform_t = self._build2dTransformMatrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)

        affine_t = F.affine_grid(transform_t[:, :2], input_g.size(), align_corners=False)
        augment_input_g = F.grid_sample(input_g, affine_t, padding_mode='border', align_corners=False)
        augment_label_g = F.grid_sample(label_g.to(torch.float32), affine_t, padding_mode='border', align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augment_input_g)
            noise_t *= self.noise
            augment_input_g += noise_t
        return augment_input_g, augment_label_g > 0.5
