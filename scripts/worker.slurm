#!/bin/sh

pwd
date

cd /home1/08115/zbian/scheduler

source ~/python-env/ones/bin/activate

ibrun python -m src.app_manager --cache_dir=$1 &

wait
