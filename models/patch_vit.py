import torch
import torch.nn as nn
import timm

class PatchSmallVit(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224",pretrained=True)
        self.vit.head = nn.Linear(384,num_classes)
    def forward(self,x):
        return self.vit(x)