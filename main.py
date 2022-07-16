import config
import torch.nn as nn
import utils
import os
import torch
import numpy as np
from models import densenet,tony_net,resnet
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
from accelerate import Accelerator
from sklearn.metrics import confusion_matrix
import random
######################################################
#                                                    #
#                                                    #
######################################################
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

seed_torch(config.parameter['seed'])

accelerator = Accelerator()

# device = torch.device(config.parameter["device"])
console.log(f"Use device {accelerator.device}")


if __name__ == "__main__":
    ## network
    network = densenet.DenseNet(config.parameter["in_channel"],
                                config.parameter["num_classes"])
    # network = network.to(device)

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
    network, optimizer, train_loader,val_loader = accelerator.prepare(network, optimizer, train_loader,val_loader)


    if not(os.path.exists("./log")):
        os.mkdir("./log")
        os.mkdir("./log/models/")

    best_acc = 0.0
    for epoch in range(config.parameter['epochs']):
        bar  =track(train_loader,description=f"[blod red]Training Epoch : {epoch + 1} ")
        for img,label in bar:
            #img = img.to(device)
            #label = label.to(device)
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
            #loss.backward()
            accelerator.backward(loss)
            optimizer.step()
        scheduler_warmup.step()

        network.eval()
        trues = []
        predicts = []
        bar = track(val_loader,description=f"[blod red]Testing Epoch : {epoch + 1} ")
        for img,label in bar:
            # img = img.to(device)
            # label = label.to(device)
            out   = network(img)
            loss  = lossfunction(out,label)
            predict =  utils.to_numpy(out.argmax(dim = -1))
            true    =  utils.to_numpy(label)
            predicts += predict.tolist()
            trues += true.tolist()

            (recall,f1,accuracy) = utils.compute_metrics(true,predict)
            val_log.update(
                recall,
                f1,
                accuracy,
                loss.item()
            )

        matrix = confusion_matrix(trues,predicts)

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
            table.add_row(*col)
        console.print(table)
        console.print("-"*30)
        console.print("confusion_matrix")
        console.print(matrix)
        console.log(f"Epoch {epoch + 1} end ...")
        network.train()
        if float(val_metrics[2]) > best_acc:
            best_acc = float(val_metrics[2])
            torch.save(network.state_dict(),f"./log/models/{config.paramters['save_model']}.pth",)

    console.log("Save Log ...")
    train_log.to_csv("./log/train_log.csv")
    val_log.to_csv("./log/val_log.csv")

            