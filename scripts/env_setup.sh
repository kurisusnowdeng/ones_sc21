#!/bin/sh

cd ~
mkdir python-env
cd python-env
virtualenv ones
source one/bin/activate
pip install --no-input torch==1.4.0 torchvision==0.5.0 --force-reinstall
pip install --no-input rpyc pytorch-pretrained-bert
deactivate
