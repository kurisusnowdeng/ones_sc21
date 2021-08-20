import argparse
import os
import random
import subprocess
import sys
import time
from threading import Thread

import numpy as np
import rpyc
import torch
from rpyc.utils.server import ThreadedServer

from .config import *
from .utils import free_port, get_local_ip, get_logger

logger = get_logger('App_Manager')


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--addr', type=str)
    parser.add_argument('--port', type=int, default=17834)
    # parser.add_argument('--controller_addr', type=str)
    parser.add_argument('--controller_port', type=int, default=34625)
    parser.add_argument('--cache_dir', type=str, default=log_path + 'cache')
    return parser.parse_args()


def new_worker(job, local_rank):
    cmd = [
        'CUDA_VISIBLE_DEVICES=' + str(local_rank),
        'NCCL_SOCKET_IFNAME=^lo,docker0', 'NCCL_DEBUG=INFO'
    ]
    cmd.extend(job['cmd'])
    cmd.extend([
        '--model_dir=' + job['model_path'],
        '--epoch_size=' + str(job['epoch_size']),
        '--batch_size=' + str(job['batch_size']), '--lr=' + str(job['lr']),
        '--local_rank=' + str(local_rank), '--job_id=' + str(job['id'])
    ])
    port = free_port()
    cmd.append('--port=' + str(port))
    return {
        'job_id': job['id'],
        'local_rank': local_rank,
        'cmd': cmd,
        'port': port
    }


class AppManager:
    def __init__(self, addr, port, ctrl_addr, ctrl_port):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.addr = addr
        self.port = port
        self.controller_addr = ctrl_addr
        self.controller_port = ctrl_port
        self.exit = False
        self.worker_list = [None] * self.num_gpus
        with rpyc.connect(ctrl_addr, ctrl_port) as conn:
            self.id = conn.root.join(addr, port, self.num_gpus)

    @property
    def num_gpus(self):
        return torch.cuda.device_count()

    def run(self, local_rank, job, resume=False, scale=False):
        logger.info('Node {}: preparing job {} on GPU {} ...'.format(
            self.id, job['id'], local_rank))
        worker = new_worker(job, local_rank)
        args = worker['cmd']
        if resume:
            args.append('--resume')
        if scale:
            args.append('--scale')
        p = subprocess.Popen(' '.join(args), shell=True)
        logger.info('Node {}: job {} is waiting for GPU {} ...'.format(
            self.id, job['id'], local_rank))

        while self.get_gpu_status(local_rank) is not None:
            logger.info('Node {} - GPU {} is not ready for job {}, used by job {}'.
                  format(self.id, local_rank, job['id'],
                         self.get_gpu_status(local_rank)))
            time.sleep(0.1)
        self.worker_list[local_rank] = worker
        self.worker_list[local_rank]['process'] = p
        logger.info('Node {}: job {} start on GPU {}.'.format(
            self.id, job['id'], local_rank))

    def setup(self, job_id, local_rank):
        with rpyc.connect(self.controller_addr, self.controller_port) as conn:
            size, rank, master_addr, master_port = conn.root.setup(
                job_id, self.id, local_rank)
        return size, rank, master_addr, master_port

    def train_ready(self, job_id, local_rank):
        worker_status = self.get_gpu_status(local_rank)
        return worker_status is not None and worker_status == job_id

    def set_scale(self, job_id, local_rank, batch_size, lr):
        logger.info('job %d: pausing node %d - GPU %d' %
                    (job_id, self.id, local_rank))
        if self.worker_list[local_rank] is not None:
            addr = 'localhost'
            port = self.worker_list[local_rank]['port']
            with rpyc.connect(addr, port) as conn:
                conn.root.set_scale(batch_size, lr)
        else:
            logger.error(
                'SetScaleError: Job {} is not on node {} - GPU {}.'.format(
                    job_id, self.id, local_rank))

    def set_stop(self, job_id, local_rank):
        logger.info('job %d: stopping node %d - GPU %d' %
                    (job_id, self.id, local_rank))
        if self.worker_list[local_rank] is not None:
            addr = 'localhost'
            port = self.worker_list[local_rank]['port']
            with rpyc.connect(addr, port) as conn:
                conn.root.set_stop()
        else:
            logger.error(
                'SetStopError: Job {} is not on node {} - GPU {}.'.format(
                    job_id, self.id, local_rank))

    def scale_ready(self, job_id, local_rank):
        with rpyc.connect(self.controller_addr, self.controller_port) as conn:
            conn.root.scale_ready(job_id, self.id, local_rank)

    def sync_progress(self, job_id):
        with rpyc.connect(self.controller_addr, self.controller_port) as conn:
            return conn.root.sync_progress(job_id)

    def update_log(self, job_id, size, rank, local_rank, epoch, num_samples,
                   batch_size, lr, throughput, loss, acc):
        with rpyc.connect(self.controller_addr, self.controller_port) as conn:
            return conn.root.update_log(job_id, size, rank, self.id,
                                        local_rank, epoch, num_samples,
                                        batch_size, lr, throughput, loss, acc)

    def broadcast_complete(self, job_id, local_rank):
        with rpyc.connect(self.controller_addr, self.controller_port) as conn:
            conn.root.broadcast_complete(job_id, self.id, local_rank)

    def worker_release(self, job_id, local_rank):
        self.worker_list[local_rank] = None

    def worker_complete(self, job_id, local_rank, save_path):
        with rpyc.connect(self.controller_addr, self.controller_port) as conn:
            conn.root.worker_complete(job_id, self.id, local_rank, save_path)
        self.worker_list[local_rank] = None

    def get_id(self):
        return self.id

    def get_gpu_status(self, local_rank):
        if self.worker_list[local_rank] is not None:
            return self.worker_list[local_rank]['job_id']
        else:
            return None

    def set_exit(self):
        self.exit = True
        return self.exit

    def terminate(self):
        if hasattr(self, 'worker_list'):
            for i, worker in enumerate(self.worker_list):
                if worker is not None:
                    if worker['process'].poll() is None:
                        worker['process'].kill()
                logger.info('Node %d - GPU %d left.' % (self.id, i))
        logger.info('Node %d left.' % self.id)


class AppService(rpyc.Service):
    def __init__(self, manager):
        super().__init__()
        self.manager = manager

    def exposed_run(self, local_rank, job, resume, scale):
        self.manager.run(local_rank, job, resume, scale)

    def exposed_setup(self, job_id, local_rank):
        return self.manager.setup(job_id, local_rank)

    def exposed_train_ready(self, job_id, local_rank):
        return self.manager.train_ready(job_id, local_rank)

    def exposed_set_scale(self, job_id, local_rank, batch_size, lr):
        self.manager.set_scale(job_id, local_rank, batch_size, lr)

    def exposed_set_stop(self, job_id, local_rank):
        self.manager.set_stop(job_id, local_rank)

    def exposed_scale_ready(self, job_id, local_rank):
        self.manager.scale_ready(job_id, local_rank)

    def exposed_sync_progress(self, job_id):
        return self.manager.sync_progress(job_id)

    def exposed_update_log(self, job_id, size, rank, local_rank, epoch,
                           num_samples, batch_size, lr, throughput, loss, acc):
        return self.manager.update_log(job_id, size, rank, local_rank, epoch,
                                       num_samples, batch_size, lr, throughput,
                                       loss, acc)

    def exposed_broadcast_complete(self, job_id, local_rank):
        self.manager.broadcast_complete(job_id, local_rank)

    def exposed_worker_release(self, job_id, local_rank):
        self.manager.worker_release(job_id, local_rank)

    def exposed_worker_complete(self, job_id, local_rank, save_path):
        self.manager.worker_complete(job_id, local_rank, save_path)

    def exposed_free_port(self):
        return free_port()

    def exposed_get_id(self):
        return self.manager.get_id()

    def exposed_get_gpu_status(self, local_rank):
        return self.manager.get_gpu_status(local_rank)

    def exposed_terminate(self):
        return self.manager.set_exit()


def main():
    ARGS = get_args()

    addr = get_local_ip()

    while not os.path.exists(ARGS.cache_dir):
        time.sleep(1)

    with open(ARGS.cache_dir, 'r') as f:
        controller_addr = f.readline().split(':')[1]

    manager = AppManager(addr, ARGS.port, controller_addr,
                         ARGS.controller_port)
    t = ThreadedServer(AppService(manager), port=ARGS.port)
    server = Thread(target=t.start, args=())
    server.daemon = True
    server.start()
    logger.info('node %d manager launched at %s:%d' %
                (manager.id, addr, ARGS.port))

    try:
        while not manager.exit:
            time.sleep(1)
    except (KeyboardInterrupt, Exception) as e:
        print(e)

    manager.terminate()
    t.close()
    sys.exit()


if __name__ == "__main__":
    main()
