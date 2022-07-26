from torchvision.models import resnet18
import config
import torch.nn as nn
import timm
class ResNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features,num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


class DenseNet(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.model = timm.create_model("densenet121", pretrained=True)
        in_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Linear(in_features,num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
