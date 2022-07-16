from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv

## simple transform convert PIL image  to tensor 
transform_tensor = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
    ]
)


### Add new transform to below
