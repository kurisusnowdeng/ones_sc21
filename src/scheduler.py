import copy
import csv
import random
import time
from abc import abstractmethod
from collections import OrderedDict, deque
from threading import Event

import numpy as np

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
        profile_path = log_path + self.name + '_profile.csv'
        with open(profile_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'submission_time', 'end_time', 'completion_time',
                'execution_time', 'waiting_time'
            ])
            for job in self.jobs:
                job_id = job['id']
                submission_time = job['start_time']
                end_time = job['end_time']
                completion_time = end_time - submission_time
                execution_time = job['run_time']
                waiting_time = completion_time - execution_time
                writer.writerow([
                    job_id,
                    time_to_str(submission_time),
                    time_to_str(end_time),
                    float_to_str(completion_time, 3),
                    float_to_str(execution_time, 3),
                    float_to_str(waiting_time, 3)
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
        self.size = K
        self.xi = xi
        self.theta = theta
        self.delta = delta
        self.monitored_jobs = self.controller.monitored_jobs
        self.monitor = self.controller.monitor

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
            if len(self._scan_active_jobs()) > 0:
                self.population = self._schedule()
                self.schedule = self._update()
            if self.iteration > 5:
                break

    def _initialize(self):
        self.iteration = 0
        population = [self._empty_schedule() for _ in range(self.size)]
        new_jobs = self._scan_new_jobs()
        for s in population:
            np.random.shuffle(new_jobs)
            for i, j in enumerate(new_jobs):
                s[i] = j
        schedule = population[0]
        for i, j in enumerate(schedule):
            if j is not None:
                node_id, local_rank = self.find_gpu(i)
                self.controller.run(j, [(node_id, local_rank)])
        logger.info('Population initialized.')
        for s in population:
            print(s)
        return population, schedule

    def _scan_executed_jobs(self):
        return [
            j for j in self._scan_active_jobs()
            if j not in self._scan_new_jobs()
        ]

    def _schedule(self):
        self.iteration += 1
        self.monitor.predict_remaining_workload(self._scan_executed_jobs())
        # scale up
        for j in self._scan_running_jobs():
            if self.jobs[j]['completed_epochs'] - self.monitored_jobs[j][
                    'last_epoch_scaled'] >= num_epochs_before_next_scaling:
                self.monitor.scale_up(j)
        # evolve
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
        for i in range(len(self.next_population)):
            cand = next_population[i]
            cand = self._pad(cand)
            next_population[i] = self._reorder(cand)
        # select
        next_population = self._select(next_population)
        return next_population

    def _get_median_run_time(self):
        run_time = list()
        cur_time = time.time()
        for job in self.jobs:
            if job['status'] == status_submitted:
                continue
            elif job['status'] == status_running:
                run_time.append(job['run_time'] + cur_time -
                                self.running_jobs[job['id']]['start_time'])
            else:
                run_time.append(job['run_time'])
        return np.median(run_time)

    def _refresh(self, schedule):
        s = self._empty_schedule()
        # remove complete jobs
        for i in range(self.cluster_size):
            j = schedule[i]
            s[i] = j
            if j is not None and self.jobs[j]['status'] != status_running:
                s[i] = None
        # find job with largest size
        largest_job = -1
        largest_job_size = 0
        for j in self._scan_running_jobs():
            if self.jobs[j]['completed_epochs'] - self.monitored_jobs[j][
                    'last_epoch_scaled'] >= num_epochs_before_next_scaling:
                job_size = self.running_jobs[j]['size']
                # if self.controller._check_worker_status(
                # 		j, [worker_running]) == job_size:
                if largest_job_size < job_size:
                    largest_job = j
                    largest_job_size = job_size
        if largest_job >= 0:
            largest_job_size = largest_job_size // 2
            run_time = self.jobs[largest_job]['run_time'] + time.time(
            ) - self.running_jobs[largest_job]['start_time']
            if run_time > self._get_median_run_time():
                # scale down
                self.monitor.scale_clear(largest_job)
                cnt = 0
                for i in range(self.cluster_size):
                    if schedule[i] == largest_job:
                        if cnt < largest_job_size:
                            cnt += 1
                        else:
                            s[i] = None
        print('Refresh:', schedule, '-->', s)
        return s

    def _get_ones_max_size(self, job_id):
        max_size = 0
        if self.jobs[job_id]['status'] == status_running:
            max_size += self.running_jobs[job_id]['size']
            if self.waiting_jobs[job_id]['status'] == wait_for_scaling:
                max_size += self.waiting_jobs[job_id]['size']
        elif self.waiting_jobs[job_id]['status'] == wait_for_scaling:
            max_size += self.waiting_jobs[job_id]['size']
        return max_size

    def _get_training_max_size(self, job_id):
        return self.monitored_jobs[job_id]['max_size']

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
                        allocation1[j2] = 0
                    if allocation1[j2] < self._get_ones_max_size(j2):
                        s2[i] = j2
                        allocation1[j2] += 1
            else:
                if j2 is not None:
                    if j2 not in allocation1:
                        allocation1[j2] = 0
                    if allocation1[j2] < self._get_ones_max_size(j2):
                        s1[i] = j2
                        allocation1[j2] += 1
                if j1 is not None:
                    if j1 not in allocation2:
                        allocation1[j1] = 0
                    if allocation1[j1] < self._get_ones_max_size(j1):
                        s2[i] = j1
                        allocation1[j1] += 1
        print('Crossover:', first, second, '-->', s1, s2)
        return s1, s2

    def _mutate(self, schedule):
        mask = dict()
        for j in schedule:
            if j not in mask:
                p = np.random.random_sample()
                if p >= self.theta:
                    mask[j] = j
                else:
                    mask[j] = None
        s = self._empty_schedule()
        for i in range(self.cluster_size):
            s[i] = mask[schedule[i]]
        print('Mutate:', schedule, '-->', s)
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
        before_pad = copy.deepcopy(allocation)
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
            j for j in self._scan_executed_jobs()
            if self.waiting_jobs[j]['size'] > 0
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
            x = self.waiting_jobs[j]['size']
            if x > C - k:
                x = C - k
            allocation[j] += x
            k += x
        # keep adding if not full
        waiting_jobs = [
            j for j in self._scan_executed_jobs() if (j not in allocation) or
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
            allocation[j] += x
            k += x
        print('Pad:', before_pad, '-->', allocation)
        return allocation

    def _reorder(self, allocation):
        s = self._empty_schedule()
        k = 0
        for j in allocation:
            for _ in range(allocation[j]):
                s[k] = j
                k += 1
        print('Reorder:', allocation, '-->', s)
        return s

    def _select(self, population):
        scores = list()
        for s in population:
            score = 0
            allocation = self._get_allocation(s)
            for j in allocation:
                y = self.jobs[j]['completed_samples']
                rho = self.monitored_jobs[j]['predicted_progress']
                n = allocation[j]
                x = self.monitor.get_throughput(j, n)
                score += y * n * (1 / (rho + 1e-12) - 1) / x
            scores.append(score)
        print('Scores:', scores)
        print('Top scores:', np.sort(scores)[:self.size])
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

    def _update(self):
        can_schedule = True
        for j in self._scan_running_jobs():
            if self.monitored_jobs[j][
                    'last_epoch_scheduled'] + update_interval >= self.jobs[j][
                        'completed_epochs']:
                can_schedule = False
                break
        if not can_schedule:
            return self.schedule
        else:
            cur_placement = self._get_placement(self.schedule)
            placement = self._get_placement(self.population[0])
            # scale down
            for j in self._scan_waiting_jobs():
                if self.waiting_jobs[j][
                        'status'] == wait_for_resuming and j not in placement:
                    self.monitor.scale_down(j)

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
                        logger.info('Scale job {0} --> {1}'.format(
                            j, placement[j]))
                        self.controller.run(j, placement[j], scale='out')
                        continue

                    scale_in = False
                    for (node_id, local_rank) in cur_placement[j]:
                        if (node_id, local_rank) not in placement[j]:
                            scale_in = True
                            break
                    if scale_in:
                        self.controller.run(j, placement[j], scale='in')

            update_complete = False
            while not update_complete:
                update_complete = True
                for j in placement:
                    if self.jobs[j]['status'] not in [
                            status_running, status_complete
                    ]:
                        update_complete = False
                time.sleep(1)
            for j in placement:
                self.monitored_jobs[j]['last_epoch_scheduled'] = self.jobs[j][
                    'completed_epochs']
            logger.info('Iteration {0}: schedule updated --> {1}'.format(
                self.iteration, self.population[0]))
            return self.population[0]
