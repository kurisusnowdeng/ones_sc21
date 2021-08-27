import csv
import time
from threading import Lock, Thread

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from torch.distributions.beta import Beta
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader, TensorDataset

from .config import *
from .utils import get_logger

logger = get_logger('Monitor')


class LogLikelihood(_Loss):
    def forward(self, x, o, y):
        a = x[:, 2]
        b = o
        y = torch.unsqueeze(y, -1)
        m = Beta(a, b)
        return torch.mean(-m.log_prob(y))


class BetaRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(predictor_input_size, predictor_hidden_size)
        self.act = nn.Tanh()
        self.output = nn.Linear(predictor_hidden_size, 1)
        self.threshold = nn.ReLU()

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param)

    def forward(self, x):
        out = self.hidden(x)
        out = self.act(out)
        out = self.output(out)
        out = self.threshold(out)
        return out + 1


class Predictor:
    def __init__(self, device):
        self.beta_regression = BetaRegression().to(device)
        self.criterion = LogLikelihood()
        self.optimizer = optim.Adam(self.beta_regression.parameters(),
                                    lr=1e-4,
                                    weight_decay=1e-4)
        self.num_epochs = predictor_train_epochs
        self.batch_size = predictor_batch_size
        self.device = device

    def fit(self, X, Y):
        if len(X) > 0:
            inputs = torch.tensor(X, dtype=torch.float)
            targets = torch.tensor(Y, dtype=torch.float)
            dataset = TensorDataset(inputs, targets)
            dataloader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True)
            final_loss = 0
            self.beta_regression.train()
            self.beta_regression.init_weights()
            for _ in range(self.num_epochs):
                total_loss = 0
                num_steps = 0
                for x, y in dataloader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    out = self.beta_regression(x)
                    loss = self.criterion(x, out, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    num_steps += 1
                final_loss = total_loss / num_steps
            logger.info('Predictor updated: loss = {:.3f}'.format(final_loss))
        else:
            logger.info('Predictor is already up to date.')

    def predict(self, X):
        self.beta_regression.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float).to(self.device)
            a = inputs[:, 2]
            b = self.beta_regression(inputs).view(-1)
            m = Beta(a, b)
            out = m.sample()
        return out.cpu().detach().tolist()


class Monitor:
    def __init__(self, jobs, running_jobs, waiting_jobs):
        self.monitored_jobs = []
        self.jobs = jobs
        self.running_jobs = running_jobs
        self.waiting_jobs = waiting_jobs
        self.predictor_ready = Lock()
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        Thread(target=self.init).start()

    def init(self):
        with self.predictor_ready:
            self.predictor = Predictor(self.device)
            self.predictor_data = list()
            with open(job_history_path, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)
                for job_id, epoch, epoch_size, num_samples, total_workload, \
                    batch_size, lr, initial_loss, loss, acc, throughput in reader:
                    if int(num_samples) < int(total_workload):
                        self.predictor_data.append([
                            int(job_id),
                            int(epoch),
                            int(epoch_size),
                            int(num_samples),
                            int(total_workload),
                            int(batch_size),
                            float(lr),
                            float(initial_loss),
                            float(loss),
                            float(acc),
                            float(throughput)
                        ])
            logger.info(
                'Job history data initialized (current length = {}).'.format(
                    len(self.predictor_data)))
            self._update_predictor()

    def _transform(self, data):
        inputs = list()
        targets = list()
        for job_id, epoch, epoch_size, num_samples, total_workload, \
            batch_size, lr, initial_loss, loss, acc, throughput in data:
            inputs.append([
                np.log(epoch_size), initial_loss, num_samples / epoch_size,
                1 - loss / initial_loss, acc
            ])
            targets.append(num_samples / total_workload)
        return inputs, targets

    def _update_predictor(self):
        if len(self.predictor_data) > predictor_epoch_size:
            data = [
                self.predictor_data[i]
                for i in np.random.choice(len(self.predictor_data),
                                          size=predictor_epoch_size,
                                          replace=False)
            ]
        else:
            data = self.predictor_data
        inputs, targets = self._transform(data)
        self.predictor.fit(inputs, targets)

    def add(self, job_id, epoch_size):
        # assert len(
        #     self.monitored_jobs
        # ) == job_id, 'MonitorAddJobError: len(jobs): %d, job id: %d' % (len(
        #     self.monitored_jobs), job_id)
        new_monitored_job = {
            'id': job_id,
            'epoch_size': epoch_size,
            'status': monitored_status_active,
            'train_log': list(),
            'initial_loss': None,
            'loss': None,
            'acc': None,
            'throughput': dict(),
            'completed_samples': 0,
            'predicted_progress': None,
            'max_size': 1,
            'last_epoch_scaled': 0,
            'last_epoch_scheduled': 0
        }
        new_monitored_job['throughput'][0] = [0.]
        self.monitored_jobs.append(new_monitored_job)

    def _add_to_history(self, job_id):
        job = self.monitored_jobs[job_id]
        job['status'] = monitored_status_complete
        with self.predictor_ready:
            for log in job['train_log']:
                if log['epoch'] > 0 and log['completed_samples'] < job['completed_samples'] \
                    and log['loss'] is not None and log['acc'] is not None:
                    data = [
                        job_id, log['epoch'], job['epoch_size'],
                        log['completed_samples'], job['completed_samples'],
                        log['batch_size'], log['lr'], job['initial_loss'],
                        log['loss'], log['acc']
                    ]
                    if log['throughput'] is None:
                        data.append(0)
                    else:
                        data.append(log['throughput'])
                    self.predictor_data.append(data)
            logger.info(
                'Job history data updated (current length = {}).'.format(
                    len(self.predictor_data)))
            self._update_predictor()

    def _predict(self, inputs):
        with self.predictor_ready:
            return self.predictor.predict(inputs)

    def predict_remaining_workload(self, job_list):
        if len(job_list) > 0:
            inputs = list()
            for job_id in job_list:
                job = self.monitored_jobs[job_id]
                inputs.append([
                    job['epoch_size'], job['initial_loss'],
                    job['completed_samples'] / job['epoch_size'],
                    1 - job['loss'] / job['initial_loss'], job['acc']
                ])
            predictions = self._predict(inputs)
            for i, progress in enumerate(predictions):
                self.monitored_jobs[
                    job_list[i]]['predicted_progress'] = progress
        else:
            logger.info('No job to predict.')

    def get_throughput(self, job_id, size):
        job = self.monitored_jobs[job_id]
        if size in job['throughput']:
            return np.mean(
                job['throughput'][size][-num_monitored_throughputs:])
        else:
            X = np.sort(list(job['throughput'].keys()), axis=None)
            Y = list(
                map(
                    lambda x: np.mean(job['throughput'][x][
                        -num_monitored_throughputs:]), X))
            f = interp1d(X, Y, bounds_error=False, fill_value='extrapolate')
            return f(size)

    def _update_progress(self, job_id, log):
        job = self.monitored_jobs[job_id]
        # record initial loss
        if log['epoch'] == 0:
            job['initial_loss'] = np.minimum(log['loss'], 100.)
        # update completed samples
        job['completed_samples'] += log['num_samples']
        # record throughput
        size = log['size']
        if log['throughput'] is not None:
            if size not in job['throughput']:
                job['throughput'][size] = list()
            job['throughput'][size].append(log['throughput'])
        # update train log
        if log['loss'] is not None:
            new_train_log = {
                'epoch': log['epoch'],
                'completed_samples': job['completed_samples'],
                'batch_size': log['batch_size'],
                'lr': log['lr'],
                'loss': log['loss'],
                'acc': log['acc'],
                'throughput': log['throughput']
            }
            job['train_log'].append(new_train_log)
            job['loss'] = log['loss']
            job['acc'] = log['acc']
        logger.info(
            'job %d - epoch %d (%d/%d samples (%d in total), batch size = %d, learning rate = %g): loss = %s, accuracy = %s, throughput = %s, best accuracy = %s'
            % (job_id, log['epoch'], log['num_samples'], job['epoch_size'],
               job['completed_samples'], log['batch_size'], log['lr'],
               "n/a" if log['loss'] is None else "%.3f" % log['loss'],
               "n/a" if log['acc'] is None else "%.3f%%" %
               (log['acc'] * 100), "n/a" if log['throughput'] is None else
               "%.3f samples/sec" % log['throughput'], "%.3f%% (%d/%d)" %
               (self.jobs[job_id]['best_acc'] * 100, self.jobs[job_id]
                ['convergence_counter'], self.jobs[job_id]['patience'])))
        return

    def scale_up(self, job_id, cluster_size):
        if self.jobs[job_id]['status'] == status_running:
            with self.waiting_jobs[job_id]['mutex']:
                if self.running_jobs[job_id]['size'] == \
                    self.monitored_jobs[job_id]['max_size']:
                    self.monitored_jobs[job_id]['max_size'] = np.minimum(
                        cluster_size,
                        self.monitored_jobs[job_id]['max_size'] * 2)
                    logger.info('Job {} scaled up: {} --> {}.'.format(
                        job_id, self.running_jobs[job_id]['size'],
                        self.monitored_jobs[job_id]['max_size']))
                self.waiting_jobs[job_id]['status'] = wait_for_scaling
                self.waiting_jobs[job_id]['size'] = np.minimum(
                    self.monitored_jobs[job_id]['max_size'],
                    self.jobs[job_id]['num_gpus'] *
                    4) - self.running_jobs[job_id]['size']

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

    def scale_down(self, job_id):
        if self.jobs[job_id]['status'] == status_running \
            and self.running_jobs[job_id]['size'] > 0:
            run_time = self.jobs[job_id]['run_time'] + time.time(
            ) - self.running_jobs[job_id]['start_time']
            median_run_time = self._get_median_run_time()
            max_size = np.maximum(
                int(2 * self.running_jobs[job_id]['size'] /
                    (scale_down_factor * run_time / median_run_time + 1)), 1)
            max_size = (max_size // 2 + max_size % 2) * 2
            # print('scaling down job {}: {} --> {}'.format(
            #     job_id, self.running_jobs[job_id]['size'], max_size))
            if run_time > median_run_time:
                with self.waiting_jobs[job_id]['mutex']:
                    self.waiting_jobs[job_id]['status'] = wait_for_scaling
                    self.waiting_jobs[job_id][
                        'size'] = max_size - self.running_jobs[job_id]['size']
                    logger.info('Job {} scaled down: {} --> {}.'.format(
                        job_id, self.running_jobs[job_id]['size'], max_size))

    def scale_in(self, job_id):
        job = self.waiting_jobs[job_id]
        if job['status'] == wait_for_resuming and job['size'] > 1:
            with job['mutex']:
                prev_size = job['size']
                job['size'] = job['size'] // 2
                self.monitored_jobs[job_id]['max_size'] = job['size']
                logger.info('Job {} scaled in: {} --> {}.'.format(
                    job_id, prev_size, job['size']))

    def complete(self):
        with open(job_history_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'id', 'epoch', 'epoch_size', 'num_samples', 'total_workload',
                'batch_size', 'lr', 'initial_loss', 'loss', 'acc', 'throughput'
            ])
            for line in self.predictor_data:
                writer.writerow(line)
        logger.info('Monitor terminated.')

    def update_progress(self, job_id, log):
        Thread(target=self._update_progress, args=(
            job_id,
            log,
        )).start()

    def add_to_history(self, job_id):
        Thread(target=self._add_to_history, args=(job_id, )).start()
