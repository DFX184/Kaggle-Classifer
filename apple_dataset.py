import torch
import pandas as pd
import config
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split,TensorDataset
from rich import print
from PIL import Image
import os
import cv2 as cv
from sklearn.preprocessing import LabelEncoder
from rich.console import Console
console = Console()
from rich.progress import track
from rich.table import Column, Table
from torchvision import transforms
from prefetch_generator import BackgroundGenerator
import jpeg4py as jpeg

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ImageDataset(Dataset):
    def __init__(self,root,df,transform = None,label_func = None):

        super().__init__()
        self.root = root
        self.images = df['image'].values.tolist()
        self.labels = df['labels'].values.tolist()
        self.transform = transform
        if not(label_func is None):
            self.labels = list(map(label_func,self.labels))
        
    def __getitem__(self,index):
        path= os.path.join(self.root,self.images[index])
        #img = cv.imread(path)
        img = jpeg.JPEG(path).decode()
        img = Image.fromarray(img)
        label = self.labels[index]
        if not(self.transform is None):
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.labels)



def create_dataloader(label_func = None,transform=None):
    path = os.path.join(config.parameter['ROOT'],
                        config.parameter['dataset_csv'])
    df = pd.read_csv(path)
    if label_func is None:
        df['labels'] = LabelEncoder().fit_transform(df['labels'])
    dataset = ImageDataset(os.path.join(config.parameter['ROOT'],
                                        "train_images"),df,transform=transform,label_func=label_func)
    print("The length of dataset is ",len(dataset))
    test_size= int(len(dataset) * config.parameter['val_size'])

    train_set,val_set = random_split(dataset,lengths = [len(dataset) - test_size,test_size],
                                        generator=torch.Generator().manual_seed(config.parameter['seed']))
    train_loader = DataLoaderX(train_set,shuffle=True,
                                batch_size =config.parameter['batch_size'],
                                num_workers = config.parameter['num_workers'],
                                pin_memory=True)
    val_loader   = DataLoaderX(val_set,shuffle=True,
                                batch_size=config.parameter['batch_size'],
                                num_workers = config.parameter['num_workers'],
                                pin_memory=True)
    return train_loader,val_loader

if __name__ == "__main__":
    param = config.parameter
    print("parameters : ",param)
    

