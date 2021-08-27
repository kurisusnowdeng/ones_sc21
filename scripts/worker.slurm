#!/bin/sh

pwd
date

cd /path/to/project

source ~/python-env/ones/bin/activate

ibrun python -m src.app_manager --cache_dir=$1 &

wait
