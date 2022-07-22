import config
import torch.nn as nn
import utils
import os
import torch
import numpy as np
from models import densenet,tony_net,resnet,effectNet,vit
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
import lossfunc
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
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

# device = torch.device(config.parameter["device"])

class LitCassava(pl.LightningModule):
    def __init__(self, model):
        super(LitCassava, self).__init__()
        self.model = model
        self.metric = pl.metrics.F1(num_classes=config.parameter['num_classes'])
        self.criterion = lossfunc.FocalLoss()
        self.lr = config.parameter['learning_rate']

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler_steplr = StepLR(self.optimizer, step_size=int(0.2 *config.parameter['epochs']), gamma=0.1)
        self.scheduler   = scheduler_warmup = GradualWarmupScheduler(
            self.optimizer, multiplier=1,
            total_epoch=config.parameter['epochs'], 
            after_scheduler=scheduler_steplr)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'train_loss': loss, 'train_f1': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch[0]
        target = batch[1]
        output = self.model(image)
        loss = self.criterion(output, target)
        score = self.metric(output.argmax(1), target)
        logs = {'valid_loss': loss, 'valid_f1': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss


if __name__ == "__main__":
    ## network
    # network = network.to(device)

    train_log = utils.ClassificationLog("Train Log")
    val_log = utils.ClassificationLog("Val Log")
    transform = vision_transforms.transform_3

    train_loader, val_loader = ap.create_dataloader(None, transform)

    model = resnet.ResNet18(config.parameter['in_channel'],config.parameter['num_classes'])

    lit_model = LitCassava(model)
    console.log("Use parameters as following:",config.parameter)
    
    logger = CSVLogger(save_dir='log/', name=config.parameter['save_model'])
    logger.log_hyperparams(config.parameter)
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                        save_top_k=1,
                                        save_last=True,
                                        save_weights_only=True,
                                        filename='checkpoint/{epoch:02d}-{valid_loss:.4f}-{valid_f1:.4f}',
                                        verbose=False,
                                        mode='min')

    trainer = Trainer(
        max_epochs=config.parameter['epochs'],
        gpus=1,
        accumulate_grad_batches=1,
        precision=16,
        checkpoint_callback=checkpoint_callback,
        logger=logger,
        weights_summary='top',
    )
    trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=val_loader)