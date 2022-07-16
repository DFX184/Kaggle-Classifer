import torch
import torch.nn.functional as F
import torch.nn as nn


class TonyNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel,32,5,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Linear(4 * 4 * 64,num_classes)
    def forward(self,x):
        x = self.layers(x)
        x = x.reshape(x.size()[0],-1)
        return self.fc(x)