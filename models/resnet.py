from torchvision.models import resnet18
import config
import torch.nn as nn
class ResNet18(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.res_net = resnet18(weights="IMAGENET1K_V1")
        self.res_net.fc = nn.Linear(512,num_classes)
    def forward(self,x):
        return self.res_net(x)
