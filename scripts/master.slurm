#!/bin/sh

pwd
date

cd /path/to/project

source ~/python-env/ones/bin/activate

python -m src.controller --size $1 --cache_dir=$2 &

python -m src.app_manager --cache_dir=$2 &

wait
