import copy
import csv
import random
import time
from abc import abstractmethod
from collections import OrderedDict, deque
from threading import Event

import numpy as np
import torch

from .config import *
from .utils import float_to_str, get_logger, time_to_str

logger = get_logger('Scheduler')


class BaseScheduler:
    def __init__(self, name, controller):
        self.name = name
        self.controller = controller
        self.cluster = self.controller.cluster
        self.jobs = self.controller.jobs
        self.running_jobs = self.controller.running_jobs
        self.waiting_jobs = self.controller.waiting_jobs
        self.exit = Event()

    def _scan_new_jobs(self):
        return self.controller.get_new_jobs()

    def _scan_active_jobs(self):
        return self.controller.get_active_jobs()

    def _scan_running_jobs(self):
        return self.controller.get_running_jobs()

    def _scan_waiting_jobs(self):
        return self.controller.get_waiting_jobs()

    def start(self):
        logger.info('Scheduler started.')
        self.exit.clear()
        self.run()
        self.exit.clear()
        self.profile()
        logger.info('Scheduler terminated.')

    @abstractmethod
    def run(self):
        pass

    def profile(self):
        profile_path = out_path + self.name + '_profile.csv'
        with open(profile_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['id', 'completion_time', 'execution_time', 'waiting_time'])
            for job in self.jobs:
                job_id = job['id']
                completion_time = end_time - submission_time
                execution_time = job['run_time']
                waiting_time = completion_time - execution_time
                writer.writerow([
                    job_id,
                    str(int(completion_time)),
                    str(int(execution_time)),
                    str(int(waiting_time))
                ])

    def wait(self, timeout):
        self.exit.wait(timeout)

    def terminate(self):
        self.exit.set()


class FCFSScheduler(BaseScheduler):
    def run(self):
        while not self.exit.is_set():
            self.wait(10)
            if len(self._scan_waiting_jobs()) > 0:
                self._schedule()

    def _schedule(self):
        free_gpus = list()
        for node_id, node in enumerate(self.cluster):
            for local_rank in range(node['num_gpus']):
                if self.controller.get_gpu_status(node_id, local_rank) is None:
                    free_gpus.append((node_id, local_rank))
        logger.info('# Free GPUs = {}'.format(len(free_gpus)))
        jobs = self._scan_new_jobs()
        logger.info('Jobs to schedule: {}'.format(
            list(
                map(lambda j: '{}({})'.format(j, self.jobs[j]['num_gpus']),
                    jobs))))
        k = 0
        for job_id in jobs:
            job = self.jobs[job_id]
            if k + job['num_gpus'] <= len(free_gpus):
                placement = free_gpus[k:k + job['num_gpus']]
                k += job['num_gpus']

                self.controller.run(job_id, placement)
                logger.info('job {} scheduled: {} GPU(s) - {}.'.format(
                    job_id, len(placement), placement))


class ONESScheduler(BaseScheduler):
    def __init__(self,
                 name,
                 controller,
                 K=population_size,
                 xi=crossover_probability,
                 theta=mutate_rate,
                 delta=update_interval):
        super().__init__(name, controller)
        assert self.controller.monitoring, \
         'ONES requires to set Controller.monitoring=True.'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.size = K
        self.xi = xi
        self.theta = theta
        self.delta = delta
        self.monitor = self.controller.monitor
        self.monitored_jobs = self.monitor.monitored_jobs

    @property
    def cluster_size(self):
        return self.controller.cluster_size()

    def find_gpu(self, x):
        return self.controller.find_gpu(x)

    def _empty_schedule(self):
        return [None for _ in range(self.cluster_size)]

    def run(self):
        self.population, self.schedule = self._initialize()
        while not self.exit.is_set():
            time.sleep(1)
            if len(self._scan_active_jobs()) > 0:
                # self.executed_jobs = list()
                self.population = self._schedule()
                self.schedule = self._update(self.schedule, self.population[0])

    def _initialize(self):
        self.iteration = 0
        population = [self._empty_schedule() for _ in range(self.size)]
        # while len(self._scan_new_jobs()) < self.size:
        while len(self._scan_new_jobs()) == 0:
            time.sleep(1)
        new_jobs = self._scan_new_jobs()
        # for _ in range(self.size - len(new_jobs)):
        #     new_jobs.append(None)

        for s in population:
            np.random.shuffle(new_jobs)
            for i, j in enumerate(new_jobs[:self.size]):
                s[i] = j
            # print(s)
        schedule = population[0]
        for i, j in enumerate(schedule):
            if j is not None:
                node_id, local_rank = self.find_gpu(i)
                self.controller.run(j, [(node_id, local_rank)])
        self._wait_for_execution(self._get_placement(population[0]))
        logger.info('Population initialized.')
        return population, schedule

    def _wait_for_execution(self, placement):
        is_complete = False
        while not is_complete:
            time.sleep(0.1)
            is_complete = True
            for j in placement:
                if not is_complete or self.jobs[j]['status'] not in [
                        status_running, status_complete
                ]:
                    is_complete = False
                    break
                elif self.jobs[j]['status'] == status_running:
                    for (node_id, local_rank) in placement[j]:
                        if (node_id, local_rank) in self.running_jobs[j]['workers'] \
                         and self.running_jobs[j]['workers'][(node_id, local_rank)]['status'] != worker_running:
                            is_complete = False
                            break
        for j in placement:
            self.monitored_jobs[j]['last_epoch_scheduled'] = self.jobs[j][
                'completed_epochs']

    def _scan_executed_jobs(self):
        return [
         j for j in self._scan_active_jobs()
         if (j not in self._scan_new_jobs()) and \
          (self.jobs[j]['completed_epochs'] >= num_epochs_before_next_scaling)
        ]

    def _schedule(self):
        self.iteration += 1
        self.executed_jobs = self._scan_executed_jobs()
        self.monitor.predict_remaining_workload(self.executed_jobs)
        # print(self.executed_jobs)
        # print([
        #  self.monitored_jobs[j]['predicted_progress']
        #  for j in self.executed_jobs
        # ])
        # scale up
        for j in self._scan_running_jobs():
            if self.jobs[j]['completed_epochs'] - self.monitored_jobs[j][
                    'last_epoch_scaled'] >= num_epochs_before_next_scaling:
                self.monitor.scale_up(j)
        # scale down
        # find job with largest size
        largest_job = -1
        largest_job_size = 0
        for j in self._scan_running_jobs():
            if self.jobs[j]['completed_epochs'] - self.monitored_jobs[j][
                    'last_epoch_scaled'] >= num_epochs_before_next_scaling:
                job_size = self.running_jobs[j]['size']
                if largest_job_size < job_size:
                    largest_job = j
                    largest_job_size = job_size

        if largest_job >= 0 and largest_job_size >= self.cluster_size // 2:
            self.monitor.scale_down(largest_job)
        # evolve
        # self.schedule = self._refresh(self.schedule)
        for i in range(self.cluster_size):
            j = self.schedule[i]
            if j is not None and self.jobs[j]['status'] == status_complete:
                self.schedule[i] = None
        next_population = list()
        for cand in self.population:
            next_population.append(self._refresh(cand))
        for _ in range(self.size // 2):
            first, second = random.sample(next_population[:self.size], 2)
            first_child, second_child = self._crossover(first, second)
            next_population.append(first_child)
            next_population.append(second_child)
        for i in range(self.size):
            cand = next_population[i]
            next_population.append(self._mutate(cand))
        for i in range(len(next_population)):
            cand = next_population[i]
            cand = self._pad(cand)
            next_population[i] = self._reorder(cand)
        # select
        next_population = self._select(next_population)
        return next_population

    def _refresh(self, schedule):
        s = self._empty_schedule()
        # remove complete jobs
        for i in range(self.cluster_size):
            j = schedule[i]
            s[i] = j
            if j is not None and self.jobs[j]['status'] == status_complete:
                s[i] = None
        allocation = dict()
        for i, j in enumerate(s):
            if j is not None:
                if j not in allocation:
                    allocation[j] = 1
                else:
                    if allocation[j] < self._get_ones_max_size(j):
                        allocation[j] += 1
                    else:
                        s[i] = None
        # print('Refresh:', schedule, '-->', s)
        return s

    def _get_ones_max_size(self, job_id):
        max_size = 0
        if self.jobs[job_id]['status'] == status_running:
            max_size += self.running_jobs[job_id]['size']
            if self.waiting_jobs[job_id]['status'] == wait_for_scaling:
                max_size += self.waiting_jobs[job_id]['size']
        elif self.waiting_jobs[job_id]['status'] == wait_for_resuming:
            max_size += self.waiting_jobs[job_id]['size']
        return int(max_size)

    def _get_training_max_size(self, job_id):
        return int(self.monitored_jobs[job_id]['max_size'])

    def _crossover(self, first, second):
        s1 = self._empty_schedule()
        s2 = self._empty_schedule()
        allocation1 = dict()
        allocation2 = dict()
        for i in range(self.cluster_size):
            j1 = first[i]
            j2 = second[i]
            p = np.random.random_sample()
            if p >= self.xi:
                if j1 is not None:
                    if j1 not in allocation1:
                        allocation1[j1] = 0
                    if allocation1[j1] < self._get_ones_max_size(j1):
                        s1[i] = j1
                        allocation1[j1] += 1
                if j2 is not None:
                    if j2 not in allocation2:
                        allocation2[j2] = 0
                    if allocation2[j2] < self._get_ones_max_size(j2):
                        s2[i] = j2
                        allocation2[j2] += 1
            else:
                if j2 is not None:
                    if j2 not in allocation1:
                        allocation1[j2] = 0
                    if allocation1[j2] < self._get_ones_max_size(j2):
                        s1[i] = j2
                        allocation1[j2] += 1
                if j1 is not None:
                    if j1 not in allocation2:
                        allocation2[j1] = 0
                    if allocation2[j1] < self._get_ones_max_size(j1):
                        s2[i] = j1
                        allocation2[j1] += 1
        # print('Crossover:', first, second, '-->', s1, s2)
        return s1, s2

    def _mutate(self, schedule):
        mask = dict()
        for j in schedule:
            if j is not None and j not in mask:
                p = np.random.random_sample()
                if self.jobs[j][
                        'completed_epochs'] < num_epochs_before_next_scaling or p >= self.theta:
                    mask[j] = j
                else:
                    mask[j] = None
        s = self._empty_schedule()
        for i in range(self.cluster_size):
            s[i] = mask[schedule[i]] if schedule[i] is not None else None
        # print('Mutate:', schedule, '-->', s)
        return s

    def _get_allocation(self, schedule):
        allocation = {}
        for j in schedule:
            if j is not None:
                if j in allocation:
                    allocation[j] += 1
                else:
                    allocation[j] = 1
        return allocation

    def _pad(self, schedule):
        allocation = self._get_allocation(schedule)
        # before_pad = copy.deepcopy(allocation)
        k = np.sum(list(allocation.values()))
        C = self.cluster_size
        # add new jobs if not full
        new_jobs = deque(
            [j for j in self._scan_new_jobs() if j not in allocation])
        while k < C and len(new_jobs) > 0:
            j = new_jobs.popleft()
            allocation[j] = 1
            k += 1
        # add waiting jobs if not full
        waiting_jobs = [
            j for j in self.executed_jobs if (j not in allocation) or (
                j in allocation and allocation[j] < self._get_ones_max_size(j))
        ]
        scores = list(
            map(
                lambda j: self.jobs[j]['completed_samples'] *
                (1 / (self.monitored_jobs[j]['predicted_progress'] + 1e-12) - 1
                 ) / self.monitor.get_throughput(j, 1), waiting_jobs))
        waiting_jobs = deque(
            list(map(lambda i: waiting_jobs[i], np.argsort(scores))))
        while k < C and len(waiting_jobs) > 0:
            j = waiting_jobs.popleft()
            x = self._get_ones_max_size(j)
            if j in allocation:
                x -= allocation[j]
            if x > C - k:
                x = C - k
            if j not in allocation:
                allocation[j] = x
            else:
                allocation[j] += x
            k += x
        # keep adding if not full
        waiting_jobs = [
            j for j in self.executed_jobs if (j not in allocation) or
            (j in allocation and allocation[j] < self._get_training_max_size(j)
             )
        ]
        scores = list(
            map(
                lambda j: self.jobs[j]['completed_samples'] *
                (1 / (self.monitored_jobs[j]['predicted_progress'] + 1e-12) - 1
                 ) / self.monitor.get_throughput(j, 1), waiting_jobs))
        waiting_jobs = deque(
            list(map(lambda i: waiting_jobs[i], np.argsort(scores))))
        while k < C and len(waiting_jobs) > 0:
            j = waiting_jobs.popleft()
            x = self._get_training_max_size(j)
            if j in allocation:
                x -= allocation[j]
            if x > C - k:
                x = C - k
            if j not in allocation:
                allocation[j] = x
            else:
                allocation[j] += x
            k += x
        # print('Pad:', schedule, '=', before_pad, '-->', allocation)
        return allocation

    def _reorder(self, allocation):
        s = self._empty_schedule()
        k = 0
        for j in self.schedule:
            if j is not None and j in allocation:
                for _ in range(int(allocation[j])):
                    s[k] = j
                    k += 1
                del allocation[j]
        for j in allocation:
            if j is not None:
                for _ in range(int(allocation[j])):
                    s[k] = j
                    k += 1
        # print('Reorder:', allocation, '-->', s)
        return s

    def _select(self, population):
        scores = list()
        for s in population:
            score = 0
            allocation = self._get_allocation(s)
            for j in allocation:
                rho = self.monitored_jobs[j]['predicted_progress']
                if rho is not None:
                    y = self.jobs[j]['completed_samples']
                    n = allocation[j]
                    x = self.monitor.get_throughput(j, n)
                    score += y * n * (1 / (rho + 1e-12) - 1) / x
            scores.append(score)
        # print('Scores:', scores)
        # print('Top scores:', np.sort(scores)[:self.size])
        selected_population = list(
            map(lambda i: population[i],
                np.argsort(scores)[:self.size]))
        return selected_population

    def _get_placement(self, schedule):
        placement = dict()
        for i, j in enumerate(schedule):
            if j is not None:
                node_id, local_rank = self.find_gpu(i)
                if j in placement:
                    placement[j].append((node_id, local_rank))
                else:
                    placement[j] = [(node_id, local_rank)]
        return placement

    def _update(self, src, dst):
        logger.info('Iteration {0}: current {1} | best {2}.'.format(self.iteration, src, dst))
        cur_placement = self._get_placement(src)
        placement = self._get_placement(dst)

        # run job immediately if all gpus are available
        all_available = dict()
        for i, j in enumerate(dst):
            if j is not None:
                if j not in all_available:
                    all_available[j] = True
                if src[i] is not None:
                    all_available[j] = False
            if src[i] is not None:
                all_available[src[i]] = False

        for j in all_available:
            if all_available[j]:
                if j not in cur_placement:
                    logger.info('Start job {0} @ {1}'.format(j, placement[j]))
                    self.controller.run(
                        j,
                        placement[j],
                        resume=(self.jobs[j]['status'] == status_stopped))
                if j in placement:
                    del placement[j]

        for j in list(cur_placement.keys()):
            if j in placement and cur_placement[j] == placement[j]:
                del cur_placement[j]
                del placement[j]

        can_update = True
        for j in cur_placement:
            if self.monitored_jobs[j][
                    'last_epoch_scheduled'] + update_interval > self.jobs[j][
                        'completed_epochs']:
                can_update = False
                break

        if not can_update:
            new_schedule = [j for j in src]
            for i, j in enumerate(dst):
                if src[i] is not None and src[i] in all_available \
                  and all_available[src[i]] and src[i] != j:
                    new_schedule[i] = None
                if j is not None and all_available[j]:
                    new_schedule[i] = j
            if new_schedule != src:
                self._wait_for_execution(self._get_placement(new_schedule))
                logger.info('Iteration {0}: schedule updated --> {1}'.format(
                    self.iteration, new_schedule))
        else:
            for j in self._scan_waiting_jobs():
                if self.waiting_jobs[j][
                        'status'] == wait_for_resuming and j not in placement:
                    self.monitor.scale_in(j)

            new_schedule = [j for j in dst]

            for j in cur_placement:
                if j not in placement:
                    logger.info('Stop job {0} @ {1}'.format(
                        j, cur_placement[j]))
                    self.controller.stop(j)

            for j in placement:
                if j not in cur_placement:
                    logger.info('Start job {0} @ {1}'.format(j, placement[j]))
                    self.controller.run(
                        j,
                        placement[j],
                        resume=(self.jobs[j]['status'] == status_stopped))
                else:
                    scale_out = False
                    for (node_id, local_rank) in placement[j]:
                        if (node_id, local_rank) not in cur_placement[j]:
                            scale_out = True
                            break
                    if scale_out:
                        logger.info('Scale job {0} @ {1} --> {2}'.format(
                            j, cur_placement[j], placement[j]))
                        self.controller.run(j, placement[j], scale='out')
                        continue

                    scale_in = False
                    for (node_id, local_rank) in cur_placement[j]:
                        if (node_id, local_rank) not in placement[j]:
                            scale_in = True
                            break
                    if scale_in:
                        logger.info('Scale job {0} @ {1} --> {2}'.format(
                            j, cur_placement[j], placement[j]))
                        self.controller.run(j, placement[j], scale='in')
            if new_schedule != src:
                self._wait_for_execution(self._get_placement(new_schedule))
                logger.info('Iteration {0}: schedule updated --> {1}'.format(
                    self.iteration, new_schedule))

        return new_schedule
