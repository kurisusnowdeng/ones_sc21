import csv
import random
import time
from abc import abstractmethod
from collections import deque
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
        jobs = self._scan_new_jobs()
        logger.info('Jobs to schedule: {}'.format(
            list(
                map(lambda j: '{}({})'.format(j, self.jobs[j]['num_gpus']),
                    jobs))))
        free_gpus = list()
        for node_id, node in enumerate(self.cluster):
            for local_rank in range(node['num_gpus']):
                if self.controller.get_gpu_status(node_id, local_rank) is None:
                    free_gpus.append((node_id, local_rank))
        k = 0
        for job_id in jobs:
            job = self.jobs[job_id]
            if k + job['num_gpus'] < len(free_gpus):
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
                 delta=update_frequency):
        super().__init__(name, controller)
        self.size = K
        self.xi = xi
        self.theta = theta
        self.delta = delta
        self.monitored_jobs = self.controller.monitored_jobs

    @property
    def cluster_size(self):
        return self.controller.cluster_size()

    def find_gpu(self, x):
        return self.controller.find_gpu(x)

    def _empty_schedule(self):
        return [None for _ in range(self.cluster_size)]

    def run(self):
        self.schedule, self.population = self._initialize()
        while not self.exit.is_set():
            if len(self._scan_waiting_jobs()) > 0:
                best_schedule, self.population = self._schedule()

    def _initialize(self):
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
        return population, schedule

    def _schedule(self):
        next_population = list()
        for cand in self.population:
            next_population.append(self._refresh(cand))
        for _ in range(self.size // 2):
            first, second = random.sample(next_population[:self.size], 2)
            first_child, second_child = self._crossover(first, second)
            next_population.append(first_child)
            next_population.append(second_child)
        for cand in next_population[:self.size]:
            next_population.append(self._mutate(cand))
        for cand in self.next_population:
            self._reorder(cand)
        best_schedule, next_population = self._select(next_population)
        return best_schedule, next_population

    def _get_allocation(self, schedule):
        allocation = {}
        for j in schedule:
            if j is not None:
                if j in allocation:
                    allocation[j] += 1
                else:
                    allocation[j] = 1
        return allocation

    def _pad(self, allocation):
        s = self._empty_schedule()
        k = 0
        for j in allocation:
            for _ in range(allocation[j]):
                s[k] = j
                k += 1
        C = len(s)
        # add new jobs if not full
        if k < C:
            new_jobs = deque([j for j in self.new_jobs if j not in allocation])
            while len(new_jobs) > 0:
                j = new_jobs.popleft()
                s[k] = j
                k += 1
                if k >= C:
                    break
        # add waiting jobs if not full
        if k < C:
            waiting_jobs = {
                j: x['size']
                for j, x in self.waiting_jobs.items()
                if j not in allocation or (
                    j in allocation and x['size'] > allocation[j])
            }
            while len(waiting_jobs) > 0:
                j, n = random.choice(list(waiting_jobs.items()))
                if n + k <= C:
                    del waiting_jobs[j]
                for _ in range(n):
                    s[k] = j
                    k += 1
                    if k >= C:
                        break
                if k >= C:
                    break
        return s

    def _refresh(self, schedule):
        allocation = {}
        # remove stopped or complete jobs
        for j in schedule:
            if j is not None and self.jobs[j]['status'] not in [
                    status_waiting, status_complete
            ]:
                if j in allocation:
                    allocation[j] += 1
                else:
                    allocation[j] = 1
        for j in allocation:
            # scale down
            allocation[j] = min(allocation[j], self.monitor.get_max_size(j))
        return self._pad(allocation)

    def _crossover(self, first, second):
        assert len(first) == len(second)
        allocation1 = {}
        allocation2 = {}
        C = len(first)
        for i in range(C):
            j1 = first[i]
            j2 = second[i]
            p = random.random()
            if p >= self.xi:
                if j1 in allocation1:
                    allocation1[j1] += 1
                else:
                    allocation1[j1] = 1
                if j2 in allocation2:
                    allocation2[j2] += 1
                else:
                    allocation2[j2] = 1
            else:
                if j2 in allocation1:
                    allocation1[j2] += 1
                else:
                    allocation1[j2] = 1
                if j1 in allocation2:
                    allocation2[j1] += 1
                else:
                    allocation2[j1] = 1
        for j in allocation1:
            allocation1[j] = min(allocation1[j],
                                 self.monitored_jobs[j]['max_size'])
        for j in allocation2:
            allocation2[j] = min(allocation2[j],
                                 self.monitored_jobs[j]['max_size'])
        s1 = self._pad(allocation1)
        s2 = self._pad(allocation2)
        assert len(s1) == len(s2)
        return s1, s2

    def _mutate(self, schedule):
        s = self._empty_schedule()
        allocation = self._get_allocation(schedule)
        for j in list(allocation.keys()):
            p = random.random()
            if p < self.theta:
                del allocation[j]
        return self._pad(allocation)

    def _reorder(self, schedule):
        s = self._empty_schedule()
        allocation = self._get_allocation(schedule)
        C = len(s)
        num_gpus = 0
        for j, n in allocation.items():
            num_gpus += n
        m = C - num_gpus
        # consolidate random placement
        k = 0
        while len(allocation) > 0:
            if m > 0:
                i = random.randint(0, m)
                m -= i
                for _ in range(i):
                    k += 1
            j, n = random.choice(list(allocation.items()))
            for _ in range(n):
                s[k] = j
                k += 1
            del allocation[j]
        return s

    def _num_nodes(self, placement):
        nodes = set()
        for node, _ in placement:
            if node not in nodes:
                nodes.add(node)
        return len(nodes)

    def _select(self, population):
        selected_population = np.random.choice(population, size=self.size)
        return selected_population

    def _update(self, schedule):
        # schedule = { job_id: placement}
        for job_id in schedule.keys():
            if self.schedule[job_id] != schedule[job_id]:
                if len(schedule[job_id]) == 0:
                    self.controller.stop(job_id)
                elif len(self.schedule[job_id]) == 0:
                    if self.jobs[job_id]['status'] == status_stopped:
                        self.controller.run(job_id,
                                            schedule[job_id],
                                            resume=True)
                    else:
                        self.controller.run(job_id, schedule[job_id])
                else:
                    self.controller.run(job_id, schedule[job_id], scale=True)
