from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import config
import numpy as np
from PIL import Image
from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations.transforms import CLAHE, GaussNoise, ISONoise
from albumentations.pytorch import ToTensorV2
import albumentations as A
## simple transform convert PIL image  to tensor 

image_size = config.parameter["image_size"]

transform_tensor = Compose([
        A.RandomResizedCrop(height=image_size[1], width=image_size[0]),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)


## Add new transform to below

### transform_1

transform_1 = Compose([
            A.RandomResizedCrop(height=image_size[1], width=image_size[0]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])


### transform_2

def histequality(img):
    for c in range(3):
        img[:,:,c] = cv.equalizeHist(img[:,:,c])
    return img

def pil_to_numpy(array):return np.asarray(array)
def numpy_to_pil(array):return Image.fromarray(array)   
def tranform_hist_equality(image,**kwargs):
    img = image
    img = histequality(img)
    return img

transform_2 = Compose([
            A.Lambda(tranform_hist_equality),
            A.RandomResizedCrop(height=image_size[1], width=image_size[0]),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
