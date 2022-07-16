from torchvision import models
from torch import nn
import torch
from rich import print

resnet50 = models.resnet50(pretrained=True)

class AvgPool(nn.Module):
    
    def forward(self, x):
        return nn.functional.avg_pool2d(x, x.shape[2:])

class ResNet50(nn.Module):
    
    def __init__(self, numclass, dropout = 0.5):
        super(ResNet50, self).__init__()
        self.resnet = resnet50
        self.resnet.layer4 = nn.Sequential(
            nn.Dropout(dropout),
            resnet50.layer4
        )
        
        self.resnet.avgpool = AvgPool()
        self.resnet.fc = nn.Linear(2048, numclass)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet(x)

if __name__ == '__main__':
    model = ResNet50(12)
    print(model)