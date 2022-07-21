from vit_pytorch import SimpleViT
import torch.nn as nn
import torch
import config
class Vit(nn.Module):
    def __init__(self,in_channel,num_classes):
        super().__init__()
        self.v = SimpleViT(
            image_size = config.parameter['image_size'][0],
            patch_size = 32,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )
    def forward(self,img):
        return self.v(img)
