import torch
import pandas as pd
import config
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from rich import print
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

class ImageDataset(Dataset):
    def __init__(self,root,df,transform = None,label_func = None):
        self.root = root
        self.images = df['image'].values.tolist()
        self.labels = df['labels'].values.tolist()
        self.transform = transform
        if not(label_func is None):
            self.labels = list(map(label_func,self.labels))
        
    def __getitem__(self,index):
        path= os.path.join(self.root,self.images[index])
        img = Image.open(path)
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
    test_size= int(len(dataset) * config.parameter['val_size'])

    train_set,val_set = random_split(dataset,lengths = [len(dataset) - test_size,test_size],
                                        generator=torch.Generator().manual_seed(config.parameter['seed']))
    train_loader = DataLoader(train_set,shuffle=True,batch_size =config.parameter['batch_size'])
    val_loader   = DataLoader(val_set,shuffle=True,batch_size=config.parameter['batch_size'])
    return train_loader,val_loader

if __name__ == "__main__":
    param = config.parameter
    print("parameters : ",param)
    

