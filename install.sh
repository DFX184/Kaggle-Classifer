#!/usr/bin/bash

echo "Installing Anaconda"

if [ !-e "Anaconda3-2022.05-Linux-x86_64.sh" ];then
    wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
fi
bash Anaconda3-2022.05-Linux-x86_64.sh
rm Anaconda3-2022.05-Linux-x86_64.sh

echo "Installing torch ..."

pip3 install torch torchvision torchaudio

echo "check cuda is available"
result=`python3 -c "import torch;print(torch.cuda.is_available())"`
if [ "$result" = "True" ];
then
    echo "CUDA is available"
else
    echo "CUDA is not available"
fi

pip install rich
pip install -U torchvision 
pip install sympy 
pip install timm 
pip install efficientnet_pytorch 
pip install -U albumentations 
pip install -U opencv-python 
pip install warmup_scheduler 
pip install prefetch_generator 
pip install jpeg4py 
sudo apt-get install libturbojpeg
pip install accelerate 
pip install -U pytorch-lightning 
pip install pytorch-lightning-bolts 
pip install wandb 
pip install gym 
