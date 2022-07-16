import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

class Bottleneck(nn.Module):
    def __init__(self, InChannel, GrowthRate):
        super(Bottleneck, self).__init__()
        self.Layer = nn.Sequential(
            nn.BatchNorm2d(InChannel),
            nn.ReLU(),
            nn.Conv2d(InChannel, GrowthRate*4, 1, bias=False),
            nn.BatchNorm2d(GrowthRate*4),
            nn.ReLU(),
            nn.Conv2d(GrowthRate*4, GrowthRate,
                      3, 1, 1, bias=False)
        )
    def forward(self, x):
        out = self.Layer(x)
        return torch.cat((x, out), dim=1)


class DenseBlock(nn.Module):
    def __init__(self, InChannel, GrowthRate, num=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(int(num)):
            layer.append(Bottleneck(InChannel, GrowthRate))
            InChannel += GrowthRate
        self.Block = nn.Sequential(*layer)

    def forward(self, x):
        return self.Block(x)


class DenseNet(nn.Module):
    def __init__(self, InChannel, numclass):
        super().__init__()
        self.module = nn.Sequential(
            DenseBlock(InChannel, 32),
            nn.MaxPool2d(2),  # 16
            nn.Dropout2d(0.2),
            DenseBlock(InChannel+32*4, 32),
            nn.MaxPool2d(2),  # 8
            nn.Dropout2d(0.2),
            DenseBlock((InChannel+32*4)+32*4, 32),
            nn.MaxPool2d(2),  # 4
            DenseBlock(((InChannel+32*4)+32*4)+32*4, 32),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, 1))

        )
        self.L = nn.Linear((((InChannel+32*4)+32*4)+32*4)+32*4, numclass)

    def forward(self, x):
        x = self.module(x)
        x = x.view(x.size()[0], -1)
        return self.L(x)

if __name__ == '__main__':
    model = DenseNet(1,1)
    print(model)