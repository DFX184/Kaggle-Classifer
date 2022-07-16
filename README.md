
# Download Dataset

if you have downloaded the dataset,the document structure should be as follows:

```
../
    ./train_images
    ./test_images
    train.csv
    sample_submission.csv
```

where `../` is `ROOT` in `config.py`

# SoftWare Version and Installation

Below are the versions of `opencv-python`, `torch`, `torchvision`, `rich`, `warmup_scheduler`, and `numpy` currently running at the time of writing this:

* `opencv-python` : 4.5.5.64 
* `torch` : 1.10.0+cu102 
* `torchvision` : 0.11.0+cu102 
* `rich` : 10.16.2  
* `warmup_scheduler` : 0.3.2 
* `numpy` : 1.20.0 

Download link:

* `opencv-python` : 4.5.5.64 
* `torch` : 1.10.0+cu102  
* `torchvision` : 0.11.0+cu102 
* `rich` : 10.16.2   
* `warmup_scheduler` : 0.3.2  
* `numpy` : 1.20.0 

Install **rich**  and **warmup_scheduler** 

```shell
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip install rich
```

# Custom Model

Add new model file to  `./models/`

**example** 

`./models/new_model.py`

```py
import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self,in_channle,num_classes):
        super().__init__()
        pass
    def forward(self):
        pass
```

# Custom transform

Add code in `vision_transform`




# Configure parameters

In `config.py`,you can see a dict  as follows:

```py
parameter = {
    "ROOT": r"J:\dataset", ## ROOT PATH
    "dataset_csv": r"train.csv",
    "batch_size": 32, 
    "device" : "cuda:0",
    "learning_rate" : 1e-3,
    "val_size" : 0.9,
    "seed" : 12,
    "epochs":10,
    "num_classes" : 12,
    "in_channel" : 3
}
```