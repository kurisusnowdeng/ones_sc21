import argparse
import csv
import math
from threading import Timer

import rpyc

from .config import *
from .utils import get_logger

cifar_script = ['python', 'examples/cifar.py', '--data_dir=$SCRATCH/cifar-10/']
imagenet_script = [
    'python', 'examples/imagenet.py', '--data_dir=$SCRATCH/imagenet_small/'
]
cola_script = [
    'python', 'examples/nlp.py', '--data_dir=$SCRATCH/glue_data/CoLA',
    '--task_name=cola', '--max_seq_length=128'
]
mrpc_script = [
    'python', 'examples/nlp.py', '--data_dir=$SCRATCH/glue_data/MRPC',
    '--task_name=mrpc', '--max_seq_length=128'
]
sst2_script = [
    'python', 'examples/nlp.py', '--data_dir=$SCRATCH/glue_data/SST-2',
    '--task_name=sst2', '--max_seq_length=128'
]

task_scripts = {
    'cifar': cifar_script,
    'imagenet': imagenet_script,
    'cola': cola_script,
    'mrpc': mrpc_script,
    'sst2': sst2_script
}

logger = get_logger('Workload')
ctrl_addr = 'localHost'
ctrl_port = 34625
trace_file = 'trace_1.csv'

def submit(model,
           task,
           epoch_size,
           patience,
           batch_size,
           lr,
           max_lr=None,
           target_acc=None,
           num_gpus=None):
    script = [cmd for cmd in task_scripts[task]]
    if model == 'bert':
        model = '$SCRATCH/bert/bert-base-uncased'
    script.append('--model=' + model)
    script.append('--patience=' + str(patience))
    with rpyc.connect(ctrl_addr, ctrl_port) as conn:
        idx = conn.root.submit(script, epoch_size, batch_size, lr, max_lr,
                               target_acc, num_gpus)
        logger.info(
            'job %d submitted: model=%s, dataset=%s(%d samples), patience=%d epochs, batch_size=%d, lr=%g, max_lr=%g, target_acc=%g%%, num_gpus=%d'
            % (idx, model.split('/')[-1], task, epoch_size, patience,
               batch_size, lr, max_lr, target_acc * 100, num_gpus))


def run_trace(N=math.inf):
    jobs = list()
    with open(trace_path+trace_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)

        for time, model, task, epoch_size, patience, batch_size, lr, max_lr, target_acc, num_gpus in reader:
            time = int(time)
            epoch_size = int(epoch_size)
            patience = int(patience)
            batch_size = int(batch_size)
            lr = float(lr)
            max_lr = float(max_lr)
            target_acc = float(target_acc)
            num_gpus = int(num_gpus)

            jobs.append(
                Timer(time,
                      submit,
                      args=[
                          model, task, epoch_size, patience, batch_size, lr,
                          max_lr, target_acc, num_gpus
                      ]))

            if len(jobs) >= N:
                break

    for t in jobs:
        t.daemon = True
        t.start()

    return len(jobs)


if __name__ == '__main__':
    run_trace(num_jobs)
