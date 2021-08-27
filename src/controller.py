import argparse
import os
import random
import sys
import time
from threading import Lock, Thread

import numpy as np
import rpyc
import torch
from rpyc.utils.server import ThreadedServer

from .config import *
from .monitor import Monitor
from .scheduler import *
from .tests.tests import run_test
from .utils import get_local_ip, get_logger
from .workload import run_trace

logger = get_logger('Controller')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=34625)
    parser.add_argument('--size', type=int)
    parser.add_argument('--scheduler',
                        type=str,
                        default='ONES',
                        help='FCFS/ONES')
    parser.add_argument('--cache_dir', type=str, default=log_path + 'cache')
    return parser.parse_args()


class Controller:
    def __init__(self, addr, port, monitoring=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.addr = addr
        self.port = port
        self.cluster = list()
        self.jobs = list()
        self.running_jobs = list()
        self.waiting_jobs = list()
        self.mutex = Lock()
        self.completed_jobs = 0
        self.monitoring = monitoring
        if monitoring:
            self.monitor = Monitor(self.jobs, self.running_jobs,
                                   self.waiting_jobs)

    def join(self, addr, port, num_gpus):
        with self.mutex:
            new_node = {
                'id': len(self.cluster),
                'addr': addr,
                'port': port,
                'num_gpus': num_gpus
            }
            self.cluster.append(new_node)
            logger.info('node %d (%s:%d) joined (# GPUs = %d)' %
                        (new_node['id'], new_node['addr'], new_node['port'],
                         new_node['num_gpus']))
        return new_node['id']

    def _create_job(self, script, epoch_size, batch_size, lr, max_lr,
                    target_acc, num_gpus):
        job_id = len(self.jobs)
        new_job = {
            'id': job_id,
            'cmd': [arg for arg in script],
            'epoch_size': epoch_size,
            'base_batch_size': batch_size,
            'batch_size': batch_size,
            'base_lr': lr,
            'lr': lr,
            'max_lr': max_lr,
            'num_gpus': num_gpus,
            'start_time': time.time(),
            'run_time': 0.0,
            'completed_epochs': 0,
            'completed_samples': 0,
            'status': None,
            'train_log': list(),
            'schedule_log': list(),
            'model_path': ckpt_path + str(job_id) + '/',
            'target_acc': target_acc,
            'best_acc': None,
            'convergence_counter': 0
            # 'patience': int(script[-1].split('=')[-1])
        }
        new_running_job = {
            'id': job_id,
            'batch_size': None,
            'lr': None,
            'start_time': None,
            'size': 0,
            'mutex': Lock(),
            'master': None,
            'placement': None,
            'workers': dict(),
            'train_log': None
        }
        new_waiting_job = {
            'id': job_id,
            'status': None,
            'mutex': Lock(),
            'size': 0
        }
        return new_job, new_running_job, new_waiting_job

    def submit(self,
               script,
               epoch_size,
               batch_size,
               lr,
               max_lr=None,
               target_acc=None,
               num_gpus=None):
        with self.mutex:
            new_job, new_running_job, new_waiting_job = self._create_job(
                script, epoch_size, batch_size, lr, max_lr, target_acc,
                num_gpus)
            new_job['status'] = status_submitted
            self.jobs.append(new_job)
            self.running_jobs.append(new_running_job)
            new_waiting_job['status'] = wait_for_start
            self.waiting_jobs.append(new_waiting_job)
            if self.monitoring:
                self.monitor.add(new_job['id'], epoch_size)
        return new_job['id']

    def _elastic_batch_size_and_lr(self, job_id, size):
        job = self.jobs[job_id]
        if job['max_lr'] is None or job['base_lr'] * size <= job['max_lr']:
            lr = job['base_lr'] * size
            batch_size = job['base_batch_size']
        else:
            lr = job['max_lr']
            batch_size = int(
                job['base_batch_size'] * lr / job['base_lr']) // size
        return batch_size, lr

    def _get_job(self, job_id):
        job = self.jobs[job_id]
        return {
            'id': job['id'],
            'cmd': job['cmd'],
            'model_path': job['model_path'],
            'epoch_size': job['epoch_size']
        }

    def _run(self, node_id, local_rank, job, resume, scale):
        with rpyc.connect(self.cluster[node_id]['addr'],
                          self.cluster[node_id]['port']) as conn:
            call = rpyc.async_(conn.root.run)
            res = call(local_rank, job, resume=resume, scale=scale)
            logger.info('job %d assigned to (%d, %d)' %
                        (job['id'], node_id, local_rank))
            while not res.ready:
                pass

    def run(self, job_id, placement, resume=False, scale=None):
        job2run = self._get_job(job_id)
        self.jobs[job_id]['status'] = status_running
        job2run['batch_size'], job2run['lr'] = self._elastic_batch_size_and_lr(
            job_id, len(placement))
        running_job = self.running_jobs[job_id]
        running_job['batch_size'] = job2run['batch_size']
        running_job['lr'] = job2run['lr']
        waiting_job = self.waiting_jobs[job_id]
        with waiting_job['mutex']:
            waiting_job['status'] = None
            waiting_job['size'] = 0
        if scale is None:
            running_job['placement'] = placement[:]
            running_job['size'] = len(placement)
            master_node, _ = placement[0]
            with rpyc.connect(self.cluster[master_node]['addr'],
                              self.cluster[master_node]['port']) as conn:
                master_addr = self.cluster[master_node]['addr']
                master_port = conn.root.free_port()
                running_job['master'] = {
                    'addr': master_addr,
                    'port': master_port
                }
            running_job['train_log'] = self._new_train_log(job_id)
            for rank, (node_id, local_rank) in enumerate(placement):
                running_job['workers'][(node_id, local_rank)] = {
                    'rank': rank,
                    'status': worker_initializing
                }
                Thread(target=self._run,
                       args=(node_id, local_rank, job2run, resume,
                             False)).start()
            running_job['start_time'] = time.time()
        elif scale == 'out':
            running_job['new_placement'] = placement[:]
            prev_size = running_job['size']
            cur_placement = running_job['placement']
            workers = running_job['workers']
            for node_id, local_rank in placement:
                if (node_id, local_rank) not in cur_placement:
                    rank = len(cur_placement)
                    cur_placement.append((node_id, local_rank))
                    workers[(node_id, local_rank)] = {
                        'rank': rank,
                        'status': worker_initializing
                    }
            running_job['size'] = len(cur_placement)
            master_node, _ = cur_placement[0]
            with rpyc.connect(self.cluster[master_node]['addr'],
                              self.cluster[master_node]['port']) as conn:
                master_addr = self.cluster[master_node]['addr']
                master_port = conn.root.free_port()
                running_job['master'] = {
                    'addr': master_addr,
                    'port': master_port
                }
            for node_id, local_rank in cur_placement[prev_size:]:
                Thread(target=self._run,
                       args=(node_id, local_rank, job2run, False,
                             True)).start()
        elif scale == 'in':
            running_job['new_placement'] = placement
            self._set_scale(job_id)

    def stop(self, job_id):
        self.jobs[job_id]['status'] = status_stopped
        with self.running_jobs[job_id]['mutex']:
            for (node_id, local_rank
                 ), worker in self.running_jobs[job_id]['workers'].items():
                if worker['status'] == worker_running:
                    with rpyc.connect(self.cluster[node_id]['addr'],
                                      self.cluster[node_id]['port']) as conn:
                        logger.info('job %d: stopping node %d - GPU %d' %
                                    (job_id, node_id, local_rank))
                        conn.root.set_stop(job_id, local_rank)

    def setup(self, job_id, node_id, local_rank):
        job = self.running_jobs[job_id]
        with job['mutex']:
            size = job['size']
            if (node_id, local_rank) in job['workers']:
                rank = job['workers'][(node_id, local_rank)]['rank']
            else:
                rank = -1
            master_addr = job['master']['addr']
            master_port = job['master']['port']
            if rank >= 0:
                logger.info('job %d: node %d - GPU %d joined (rank %d/%d)' %
                            (job_id, node_id, local_rank, rank, size))
                if job['workers'][(node_id, local_rank)]['status'] in [
                        worker_initializing, worker_scale_complete
                ]:
                    job['workers'][(node_id,
                                    local_rank)]['status'] = worker_running
            else:
                logger.info('job %d: node %d - GPU %d released' %
                            (job_id, node_id, local_rank))
        return size, rank, master_addr, master_port

    def _set_scale(self, job_id):
        with self.running_jobs[job_id]['mutex']:
            for (node_id, local_rank
                 ), worker in self.running_jobs[job_id]['workers'].items():
                if worker['status'] == worker_running:
                    logger.info('job %d: pausing node %d - GPU %d' %
                                (job_id, node_id, local_rank))
                    with rpyc.connect(self.cluster[node_id]['addr'],
                                      self.cluster[node_id]['port']) as conn:
                        conn.root.set_scale(
                            job_id, local_rank,
                            self.running_jobs[job_id]['batch_size'],
                            self.running_jobs[job_id]['lr'])

    def _check_worker_status(self, job_id, status_list):
        cnt = 0
        for _, worker in self.running_jobs[job_id]['workers'].items():
            if worker['status'] in status_list:
                cnt += 1
        return cnt

    def scale_ready(self, job_id, node_id, local_rank):
        job = self.running_jobs[job_id]
        workers = job['workers']
        if (node_id, local_rank) in workers:
            if workers[(node_id, local_rank)]['status'] == worker_initializing:
                workers[(node_id, local_rank)]['status'] = worker_scale_ready
                if self._check_worker_status(
                        job_id,
                    [worker_running, worker_scale_ready]) == job['size']:
                    self._set_scale(job_id)
            elif workers[(node_id, local_rank)]['status'] == worker_running:
                workers[(node_id, local_rank)]['status'] = worker_scale_ready
        else:
            logger.error('Node {} - GPU {} is not assigned to job {}.'.format(
                node_id, local_rank, job_id))

    def sync_progress(self, job_id):
        job = self.jobs[job_id]
        return job['completed_epochs'], job['lr'], job['current_loss'], job[
            'current_acc'], job['convergence_counter']

    def broadcast_complete(self, job_id, node_id, local_rank):
        job = self.running_jobs[job_id]
        with job['mutex']:
            job['workers'][(node_id,
                            local_rank)]['status'] = worker_scale_complete
            if self._check_worker_status(
                    job_id, [worker_scale_complete]) == job['size']:
                job['placement'] = job['new_placement'][:]
                del job['new_placement']
                job['size'] = len(job['placement'])
                master_node, _ = job['placement'][0]
                with rpyc.connect(self.cluster[master_node]['addr'],
                                  self.cluster[master_node]['port']) as conn:
                    master_addr = self.cluster[master_node]['addr']
                    master_port = conn.root.free_port()
                    job['master'] = {'addr': master_addr, 'port': master_port}
                job['workers'] = {(i, j): {
                    'rank': rank,
                    'status': worker_scale_complete
                }
                                  for rank, (i,
                                             j) in enumerate(job['placement'])}
                logger.info('job %d: scaling complete --> ' % job_id +
                            str(job['placement']))
                if self.monitoring:
                    self.monitor.monitored_jobs[job_id][
                        'last_scaled_epoch'] = self.jobs[job_id][
                            'completed_epochs']
        return

    def worker_complete(self, job_id, node_id, local_rank, save_path):
        self.running_jobs[job_id]['workers'][(
            node_id, local_rank)]['status'] = worker_complete
        job = self.jobs[job_id]
        if save_path is not None:  # master worker, rank 0
            job['model_path'] = save_path
        with self.running_jobs[job_id]['mutex']:
            if self._check_worker_status(
                    job_id,
                [worker_complete
                 ]) == self.running_jobs[job_id]['size']:  # all workers end
                end_time = time.time()
                job['run_time'] += end_time - self.running_jobs[job_id][
                    'start_time']
                if job['status'] == status_running:
                    if self.monitoring:
                        self.monitor.add_to_history(job_id)
                    job['end_time'] = end_time
                    job['completion_time'] = end_time - job['start_time']
                    logger.info('job %d complete, %d samples completed' %
                                (job_id, job['completed_samples']))
                    job['status'] = status_complete
                    self.completed_jobs += 1
                elif job['status'] == status_stopped:
                    # move job from running_jobs to waiting_jobs
                    waiting_job = self.waiting_jobs[job_id]
                    with waiting_job['mutex']:
                        waiting_job['status'] = wait_for_resuming
                        if self.monitoring:
                            waiting_job['size'] = self.monitor.monitored_jobs[
                                job_id]['max_size']
                        else:
                            waiting_job['size'] = self.running_jobs[job_id][
                                'size']
                    logger.info('job %d stopped, %d samples completed' %
                                (job_id, job['completed_samples']))
                self.running_jobs[job_id]['workers'].clear()
        return

    def _new_train_log(self, job_id):
        return {
            'id': len(self.jobs[job_id]['train_log']),
            'submitted_workers': 0,
            'num_samples': 0,
            'batch_size': 0,
            'loss': 0.,
            'acc': 0.,
            'best_acc': 0.,
            'throughput': 0.,
            'placement': []
        }

    def _early_stop(self, job_id, acc):
        job = self.jobs[job_id]
        early_stop = False
        if job['target_acc'] is not None and \
         acc >= job['target_acc']:
            early_stop = True
        if job['best_acc'] is not None and \
         acc < job['best_acc'] + convergence_delta:
            early_stop = True
        else:
            job['best_acc'] = acc
        if early_stop:
            job['convergence_counter'] += 1
        else:
            job['convergence_counter'] = 0

    def update_log(self, job_id, size, rank, node_id, local_rank, epoch,
                   num_samples, batch_size, lr, throughput, loss, acc):
        job = self.running_jobs[job_id]
        log = job['train_log']
        with job['mutex']:
            log['batch_size'] += batch_size
            log['num_samples'] += num_samples
            if loss is not None:
                log['loss'] += loss
            else:
                log['loss'] = None
            if acc is not None:
                log['acc'] += acc
            else:
                log['acc'] = None
            if throughput is not None:
                log['throughput'] += throughput
            else:
                log['throughput'] = None
            log['placement'].append(tuple((node_id, local_rank)))
            log['submitted_workers'] += 1
            if log['submitted_workers'] == size:
                log['size'] = size
                log['epoch'] = epoch
                self.jobs[job_id]['batch_size'] = log['batch_size']
                self.jobs[job_id]['lr'] = log['lr'] = lr
                if log['loss'] is not None and log['acc'] is not None:
                    log['loss'] /= log['size']
                    self.jobs[job_id]['current_loss'] = log['loss']
                    log['acc'] /= log['size']
                    self.jobs[job_id]['current_acc'] = log['acc']
                    self.jobs[job_id]['completed_epochs'] += 1
                    self._early_stop(job_id, log['acc'])
                self.jobs[job_id]['completed_samples'] += log['num_samples']
                self.jobs[job_id]['train_log'].append(log)
                if self.monitoring:
                    self.monitor.update_progress(job_id, log)
                job['train_log'] = self._new_train_log(job_id)
        return

    def get_new_jobs(self):
        return [
            job['id'] for job in self.jobs
            if (job['status'] == status_submitted) or (
                job['status'] == status_stopped
                and self.waiting_jobs[job['id']]['status'] == wait_for_resuming
                and job['completed_epochs'] < num_epochs_before_next_scaling)
        ]

    def get_running_jobs(self):
        return [
            job['id'] for job in self.jobs if job['status'] == status_running
        ]

    def get_waiting_jobs(self):
        return [
            job['id'] for job in self.waiting_jobs if job['status'] is not None
            and self.jobs[job['id']] != status_complete
        ]

    def get_active_jobs(self):
        return [
            job['id'] for job in self.jobs if job['status'] != status_complete
        ]

    def terminate(self, path):
        for node in self.cluster:
            with rpyc.connect(node['addr'], node['port']) as conn:
                conn.root.terminate()
        if self.monitoring:
            self.monitor.complete()
        os.remove(path)

    def get_gpu_status(self, node_id, local_rank):
        with rpyc.connect(self.cluster[node_id]['addr'],
                          self.cluster[node_id]['port']) as conn:
            return conn.root.get_gpu_status(local_rank)

    def print_cluster(self):
        logger.info('Checking cluster status ...')
        for node in self.cluster:
            with rpyc.connect(node['addr'], node['port']) as conn:
                for local_rank in range(node['num_gpus']):
                    job_id = conn.root.get_gpu_status(local_rank)
                    if job_id is None:
                        logger.info(
                            'Node {} - GPU {} is running job {}.'.format(
                                node['id'], local_rank, job_id))
                    else:
                        epoch = self.jobs[job_id]['completed_epochs']
                        if self.monitoring:
                            last_scheduled_epoch = self.monitor.monitored_jobs[
                                job_id]['last_epoch_scheduled']
                            logger.info(
                                'Node {} - GPU {} is running job {} @ epoch {} (+{}).'
                                .format(node['id'], local_rank, job_id, epoch,
                                        epoch - last_scheduled_epoch))
                        else:
                            logger.info(
                                'Node {} - GPU {} is running job {} @ epoch {}.'
                                .format(node['id'], local_rank, job_id, epoch))

    def num_free_gpus(self):
        num_free_gpus = 0
        for node in self.cluster:
            with rpyc.connect(node['addr'], node['port']) as conn:
                for local_rank in range(node['num_gpus']):
                    if conn.root.get_gpu_status(local_rank) is None:
                        num_free_gpus += 1
        return num_free_gpus

    def cluster_size(self):
        return sum(map(lambda x: x['num_gpus'], self.cluster))

    def find_gpu(self, x):
        for node in self.cluster:
            if x < node['num_gpus']:
                return node['id'], x
            else:
                x -= node['num_gpus']

    # def _get_node_size(self):
    #     return max(map(lambda x: x['num_gpus'], self.cluster))


class ControllerService(rpyc.Service):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def exposed_join(self, addr, port, num_gpus):
        return self.controller.join(addr, port, num_gpus)

    def exposed_submit(self, script, epoch_size, batch_size, lr, max_lr,
                       target_acc, num_gpus):
        return self.controller.submit(script, epoch_size, batch_size, lr,
                                      max_lr, target_acc, num_gpus)

    def exposed_setup(self, job_id, node_id, local_rank):
        return self.controller.setup(job_id, node_id, local_rank)

    def exposed_scale_ready(self, job_id, node_id, local_rank):
        self.controller.scale_ready(job_id, node_id, local_rank)

    def exposed_sync_progress(self, job_id):
        return self.controller.sync_progress(job_id)

    def exposed_broadcast_complete(self, job_id, node_id, local_rank):
        Thread(target=self.controller.broadcast_complete,
               args=(
                   job_id,
                   node_id,
                   local_rank,
               )).start()

    def exposed_worker_complete(self, job_id, node_id, local_rank, save_path):
        Thread(target=self.controller.worker_complete,
               args=(
                   job_id,
                   node_id,
                   local_rank,
                   save_path,
               )).start()

    def exposed_update_log(self, job_id, size, rank, node_id, local_rank,
                           epoch, num_samples, batch_size, lr, throughput,
                           loss, acc):
        return self.controller.update_log(job_id, size, rank, node_id,
                                          local_rank, epoch, num_samples,
                                          batch_size, lr, throughput, loss,
                                          acc)


scheduler_dict = {'FCFS': FCFSScheduler, 'ONES': ONESScheduler}


def run_scheduler(controller, scheduler='FCFS'):
    s = scheduler_dict[scheduler](scheduler, controller)
    t = Thread(target=s.start, args=())
    t.start()
    return s, t


def terminate_scheduler(scheduler, thread):
    scheduler.terminate()
    thread.join()


def main():
    ARGS = get_args()

    if not os.path.exists(trace_path):
        os.makedirs(trace_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    controller = Controller(get_local_ip(),
                            ARGS.port,
                            monitoring=ARGS.scheduler == 'ONES')

    t = ThreadedServer(ControllerService(controller), port=controller.port)
    server = Thread(target=t.start, args=())
    server.daemon = True
    server.start()

    with open(ARGS.cache_dir, 'w') as f:
        f.write('controller_ip:' + controller.addr)

    logger.info('Controller launched at %s:%d' %
                (controller.addr, controller.port))

    while len(controller.cluster) < ARGS.size:
        time.sleep(1)

    logger.info('All nodes joined (%d/%d)' %
                (len(controller.cluster), ARGS.size))

    sched, t_sched = run_scheduler(controller, ARGS.scheduler)
    num_submitted_jobs = run_trace(num_jobs)
    # num_submitted_jobs = run_test(controller)

    try:
        while controller.completed_jobs < num_submitted_jobs:
            time.sleep(10)
            controller.print_cluster()
    except (KeyboardInterrupt, Exception) as e:
        print(e)

    controller.terminate(ARGS.cache_dir)
    terminate_scheduler(sched, t_sched)
    logger.info('System shut down.')
    sys.exit()


if __name__ == '__main__':
    main()
