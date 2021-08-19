#!/bin/sh

mkdir log

mkdir log/test

mkdir log/slurm

mkdir checkpoints

mkdir checkpoints/test

cd $SCRATCH

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zLi5BKEH1TEbkPriLAJmW4seaq7Tx254' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zLi5BKEH1TEbkPriLAJmW4seaq7Tx254" -O cifar-10.tar.gz && rm -rf /tmp/cookies.txt

tar zxvf cifar-10.tar.gz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1naWkJtoBZKv2nF_BKRG64Dumso3nXxrN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1naWkJtoBZKv2nF_BKRG64Dumso3nXxrN" -O glue_data.tar.gz && rm -rf /tmp/cookies.txt

tar zxvf glue_data.tar.gz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fGKXuNGfgd7ErHew3HpFfp6skm4P1yIR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fGKXuNGfgd7ErHew3HpFfp6skm4P1yIR" -O imagenet_small.tar.gz && rm -rf /tmp/cookies.txt

tar zxvf imagenet_small.tar.gz

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mvNOOS2b4X6FG76wrpxTBj-dcSyYHCg5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mvNOOS2b4X6FG76wrpxTBj-dcSyYHCg5" -O bert.tar.gz && rm -rf /tmp/cookies.txt

tar zxvf bert.tar.gz