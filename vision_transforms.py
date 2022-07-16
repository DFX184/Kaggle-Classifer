from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import config
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import CLAHE, GaussNoise, ISONoise
from albumentations.pytorch import ToTensorV2
import albumentations as A
## simple transform convert PIL image  to tensor 
transform_tensor = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
    ]
)


## Add new transform to below


### transform_1
image_size = config.parameter["image_size"]
transform_1 = Compose([
            A.RandomResizedCrop(height=image_size[1], width=image_size[0]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])