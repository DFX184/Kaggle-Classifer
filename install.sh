#!/usr/bin/bash

echo "Installing Anaconda"

wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
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


pip3 install opencv-python
pip3 install rich
pip3 install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
pip3 install accelerate
pip3 install prefetch_generator 
pip3 install -U albumentations