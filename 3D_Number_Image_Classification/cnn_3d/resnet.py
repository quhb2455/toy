import torch
from torch import nn
from torch.nn import functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True, **kwargs):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, **kwargs)
        self.bn = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, stride=1, **kwargs):
        super(BottleNeck, self).__init__()
        self.resblock = nn.Sequential(
            Conv_Block(in_ch, mid_ch, kernel_size=1, stride=1),
            Conv_Block(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1),
            Conv_Block(mid_ch, out_ch, activation=False, kernel_size=1, stride=1)
        )
        self.relu = nn.ReLU()
        self.skip = nn.Sequential()

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        x = self.resblock(x) + self.skip(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self,
                 num_layers=[3, 4, 6, 3],
                 in_channels=[64, 256, 512, 1024],
                 stride=[1, 2, 2, 2],
                 num_classes=1000):

        super().__init__()

        layers = self.create_layers(num_layers=num_layers,
                                    in_channels=in_channels,
                                    stride=stride)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(*layers[0])
        self.conv3 = nn.Sequential(*layers[1])
        self.conv4 = nn.Sequential(*layers[2])
        self.conv5 = nn.Sequential(*layers[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_channels[-1] * 2, num_classes)

    def create_layers(self,
                      num_layers=[3, 4, 6, 3],
                      in_channels=[64, 256, 512, 1024],
                      stride=[1, 2, 2, 2]):
        layers = []
        for idx, num in enumerate(num_layers):
            seq = []
            in_c = in_channels[idx]
            mid_c = max(in_channels[0], in_c // 2)
            out_c = in_c * 2 if idx != 0 else in_c * 4

            for repeat in range(num):

                if repeat == 0:
                    seq.append(BottleNeck(in_c, mid_c, out_c, stride=stride[idx]))

                else:
                    in_c = out_c
                    seq.append(BottleNeck(in_c, mid_c, out_c))
            layers.append(seq)
        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x