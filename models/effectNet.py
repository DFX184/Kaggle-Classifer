import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EffectNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b1',num_classes=num_classes)
    def forward(self,x):
        return self.net(x)