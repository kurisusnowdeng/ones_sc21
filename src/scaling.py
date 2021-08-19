import time
from threading import Thread

import rpyc
import torch
import torch.distributed as dist
from rpyc.utils.server import ThreadedServer

from .config import *
from .utils import get_logger


class DistService(rpyc.Service):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def exposed_set_scale(self, batch_size, lr):
        self.agent.set_scale(batch_size, lr)

    def exposed_set_stop(self):
        self.agent.set_stop()

    # def exposed_adjust_lr(self, lr):
    #     self.agent.adjust_learning_rate(lr)


def init(port, agent):
    svc = DistService(agent)
    server = ThreadedServer(svc, port=port)
    t = Thread(target=server.start, args=())
    t.daemon = True
    t.start()
    return server


class ScalingAgent:
    def __init__(self,
                 job_id,
                 model_name,
                 local_rank,
                 port,
                 mgr_addr='localhost',
                 mgr_port=17834):
        self.job_id = job_id
        self.name = model_name
        # communication settings
        self.size = None
        self.rank = None
        self.local_rank = None
        self.master_addr = None
        self.master_port = None
        self.local_rank = local_rank
        # manager server connection
        self.manager_addr = mgr_addr
        self.manager_port = mgr_port
        # signals
        self.flag_scale = False
        self.flag_stop = False
        self.exit_status = None
        with rpyc.connect(mgr_addr, mgr_port) as conn:
            self.node_id = conn.root.get_id()
        self.logger = get_logger('Worker_' + str(self.job_id) + '_' +
                                 str(self.node_id) + '_' + str(local_rank))
        self.service = init(port, self)
        self.logger.info('Job {} (node {} - GPU {}) started.'.format(
            self.job_id, self.node_id, self.local_rank))

    def load(
            self,
            net,
            criterion,
            optimizer,
            trainset,
            testset,
            batch_size,
            lr,
            num_labels,
            start_epoch=0,
            #  num_samples=0,
            scale=False,
            device='cuda'):
        # load distributed model
        self.device = device
        self.batch_size = batch_size
        self.scaled_lr = lr
        self.lr = lr
        self.start_epoch = start_epoch
        # self.num_samples = num_samples
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainset = trainset
        self.testset = testset
        self.epoch_size = len(trainset)
        self.num_labels = num_labels
        self.convergence_counter = 0

        self.logger.info('Job {} - GPU {} initialization complete.'.format(
            self.job_id, self.local_rank))

        if scale:
            self._scale_sync()
            self.start_epoch, self.lr, _, _, _ = self._sync_progress()

        self._setup()

        while not self.train_ready():
            self.logger.info('Job {}: GPU {} is not ready'.format(
                self.job_id, self.local_rank))
            time.sleep(0.1)

        self.dist_net = torch.nn.parallel.DistributedDataParallel(self.net)
        self.trainloader = self._dist_loader(self.trainset)
        self.adjust_learning_rate()
        self.logger.info(
            'Job {} (node {} - GPU {}): scaling agent initialized.'.format(
                self.job_id, self.node_id, self.local_rank))

    def get_lr(self):
        return self.lr

    def _dist_loader(self, dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.size, rank=self.rank)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           sampler=sampler,
                                           num_workers=8)

    def _setup(self):
        with rpyc.connect(self.manager_addr, self.manager_port) as conn:
            self.size, self.rank, self.master_addr, self.master_port = conn.root.setup(
                self.job_id, self.local_rank)
        self.logger.info('Setting up job {} ({}/{}) on node {} - GPU {} ...'.format(
            self.job_id, self.rank, self.size, self.node_id, self.local_rank))
        if self.rank >= 0:
            dist.init_process_group(backend="nccl",
                                    init_method="tcp://" + self.master_addr +
                                    ":" + str(self.master_port),
                                    world_size=self.size,
                                    rank=self.rank)
            self.logger.info('GPU %d : NCCL setup complete' % self.local_rank)

    def _sync_progress(self):
        with rpyc.connect(self.manager_addr, self.manager_port) as conn:
            return conn.root.sync_progress(self.job_id)

    def _scale_sync(self):
        with rpyc.connect(self.manager_addr, self.manager_port) as conn:
            conn.root.scale_ready(self.job_id, self.local_rank)
            self.size, self.rank, self.master_addr, self.master_port = conn.root.setup(
                self.job_id, self.local_rank)
        if self.rank >= 0:
            dist.init_process_group(backend="nccl",
                                    init_method="tcp://" + self.master_addr +
                                    ":" + str(self.master_port),
                                    world_size=self.size,
                                    rank=self.rank)
            for param in self.net.parameters():
                dist.broadcast(param.data, src=0)
            with rpyc.connect(self.manager_addr, self.manager_port) as conn:
                conn.root.broadcast_complete(self.job_id, self.local_rank)
            dist.barrier()
            dist.destroy_process_group()

    def train_ready(self):
        with rpyc.connect(self.manager_addr, self.manager_port) as conn:
            return conn.root.train_ready(self.job_id, self.local_rank)

    def set_scale(self, batch_size, lr):
        if not self.flag_scale:
            self.batch_size = batch_size
            self.scaled_lr = lr
            self.flag_scale = True

    def set_stop(self):
        if not self.flag_stop:
            self.exit_status = exit_stopped
            self.flag_stop = True

    def check_pause(self):
        # pause when all workers are set
        pause = torch.zeros(1) \
            if self.flag_scale or self.flag_stop \
                else torch.ones(1)
        pause = pause.to(self.device)
        dist.all_reduce(pause)
        return pause == 0

    def scale(self):
        dist.barrier()
        dist.destroy_process_group()
        self._scale_sync()
        self._setup()
        if self.rank < 0:
            # release this worker
            self.exit_status = exit_released
        else:
            self.dist_net = torch.nn.parallel.DistributedDataParallel(self.net)
            self.trainloader = self._dist_loader(self.trainset)
        # scaling complete
        self.flag_scale = False

    def upload_log(self,
                   epoch,
                   num_samples,
                   throughput=None,
                   loss=None,
                   acc=None):
        self.logger.info('Job {}: finishing epoch {} ...'.format(
            self.job_id, epoch))
        with rpyc.connect(self.manager_addr, self.manager_port) as conn:
            conn.root.update_log(self.job_id, self.size, self.rank,
                                 self.local_rank, epoch, num_samples,
                                 self.batch_size, self.lr, throughput, loss,
                                 acc)
        dist.barrier()
        epoch, _, loss, acc, convergence_counter = self._sync_progress()
        self.convergence_counter = convergence_counter
        return epoch

    def complete(self, save_path=None):
        with rpyc.connect(self.manager_addr, self.manager_port) as conn:
            if self.exit_status in [exit_complete, exit_stopped]:
                dist.barrier()
                conn.root.worker_complete(self.job_id, self.local_rank,
                                          save_path)
            else:
                conn.root.worker_release(self.job_id, self.local_rank)
        self.service.close()
        self.logger.info('Job {} (node {} - GPU {}) finished.'.format(
            self.job_id, self.node_id, self.local_rank))

    def adjust_learning_rate(self, lr=None):
        if lr is None:
            lr = self.lr
        else:
            self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
