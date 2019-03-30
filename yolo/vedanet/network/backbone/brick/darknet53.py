import os
from collections import OrderedDict
import torch
import torch.nn as nn

from ... import layer as vn_layer


class StageBlock(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
                    vn_layer.Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1),
                    vn_layer.Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 1)
                )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):
    custom_layers = (StageBlock, StageBlock.custom_layers)
    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(vn_layer.Conv2dBatchLeaky(nchannels, 2*nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(StageBlock(2*nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


class HeadBody(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels/2)
        else:
            half_nchannels = int(nchannels/3)
        in_nchannels = 2 * half_nchannels
        layers = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
                ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x

class HeadBody_SPP(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels/2)
        else:
            half_nchannels = int(nchannels/3)
        in_nchannels = 2 * half_nchannels
        layers_0 = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
                ]
        self.maxpool_0 = nn.MaxPool2d(5, stride=(1,1), padding=(2,2))
        self.maxpool_1 = nn.MaxPool2d(9, stride=(1,1), padding=(4,4))
        self.maxpool_2 = nn.MaxPool2d(13,stride=(1,1), padding=(6,6))
        layers_1 = [
                vn_layer.Conv2dBatchLeaky(half_nchannels*4,half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
                ]
        self.feature_0 = nn.Sequential(*layers_0)
        self.feature_1 = nn.Sequential(*layers_1)

    def forward(self, data):
        feature_0 = self.feature_0(data)
        mp_0 = self.maxpool_0(feature_0)
        mp_1 = self.maxpool_1(feature_0)
        mp_2 = self.maxpool_2(feature_0)
        feature_1 = torch.cat((feature_0, mp_0, mp_1, mp_2),1)
        feature_1 = self.feature_1(feature_1)
        return feature_1

class Transition(nn.Module):
    custom_layers = ()
    def __init__(self, nchannels):
        super().__init__()
        half_nchannels = int(nchannels/2)
        layers = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                nn.Upsample(scale_factor=2)
                ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return x

