import config
import torch.nn as nn
import utils
import torch
import numpy as np
from models import densenet
from rich import print
import vision_transforms
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from warmup_scheduler import GradualWarmupScheduler
import apple_dataset as ap
from rich.columns import Columns
from rich.console import Console
console = Console()
from rich.progress import track
from rich.table import Column, Table

device = torch.device(config.parameter["device"])
console.log(f"Use device {device}")

## network
network = densenet.DenseNet(config.parameter["in_channel"],
                            config.parameter["num_classes"])
network = network.to(device)

train_log = utils.ClassificationLog("Train Log")
val_log = utils.ClassificationLog("Val Log")
transform = vision_transforms.transform_tensor

train_loader, val_loader = ap.create_dataloader(None, transform)
console.log("Use parameters as following:",config.parameter)
lossfunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    network.parameters(), lr=config.parameter['learning_rate'])
scheduler_steplr = StepLR(optimizer, step_size=int(0.2 * config.parameter['epochs']), gamma=0.1)
scheduler_warmup = GradualWarmupScheduler(
    optimizer, multiplier=1,
    total_epoch=config.parameter['epochs'], 
    after_scheduler=scheduler_steplr)



for epoch in range(config.parameter['epochs']):
    
    for img,label in track(train_loader,description=f"[blod red]Training Epoch : {epoch + 1} "):
        img = img.to(device)
        label = label.to(device)
        out   = network(img)
        loss  = lossfunction(out,label)
        predict =  utils.to_numpy(out.argmax(dim = -1))
        true    =  utils.to_numpy(label)
        
        (recall,f1,accuracy) = utils.compute_metrics(true,predict)
        train_log.update(
            recall,
            f1,
            accuracy,
            loss.item()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler_warmup.step()

    network.eval()
    for img,label in track(val_loader,description=f"[blod red]Testing Epoch : {epoch + 1} "):
        img = img.to(device)
        label = label.to(device)
        out   = network(img)
        loss  = lossfunction(out,label)
        predict =  utils.to_numpy(out.argmax(dim = -1))
        true    =  utils.to_numpy(label)

        (recall,f1,accuracy) = utils.compute_metrics(true,predict)
        val_log.update(
            recall,
            f1,
            accuracy,
            loss.item()
        )

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metrics", style="dim", width=12)
    table.add_column(f"Train (Epoch = {epoch + 1})",justify="right")
    table.add_column(f"Val (Epoch = {epoch + 1})",justify="right")

    val_metrics = val_log.Average()
    train_metrics= train_log.Average()
    val_metrics = list(map(lambda x : str(round(x,5)),val_metrics))
    train_metrics = list(map(lambda x : str(round(x,5)),train_metrics))
    metrics_name  = ['Recall',"F1-score","Accuracy","Loss"]

    for col in zip(metrics_name,train_metrics,val_metrics):
        col = list(col)
        table.add_row(col)
    console.print(table)
    console.log(f"Epoch {epoch + 1} end ...")
    network.train()

        