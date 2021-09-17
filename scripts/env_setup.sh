#!/bin/sh

cd ~
mkdir python-env
cd python-env
virtualenv ones
source ones/bin/activate
pip install --no-input torch==1.4.0 torchvision==0.5.0 --force-reinstall
pip install --no-input rpyc pytorch-pretrained-bert matplotlib requests
pip install --no-input scipy==1.3.1 --force-reinstall
deactivate
