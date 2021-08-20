import os
import time

import torch

from .config import *


def _train_epoch(sa, epoch):
    # sa.logger.info('Number of workers: %d' % sa.size)
    global_batch_size = sa.size * sa.batch_size
    sa.logger.info('Job %d - Epoch %d start: lr = %g' %
                   (sa.job_id, epoch, sa.scaled_lr))
    sa.dist_net.train()

    # warm up lr if lr is scaled
    cur_lr = sa.get_lr()
    warm_up = False
    if abs(cur_lr - sa.scaled_lr) > 1e-6:
        warm_up = True
        lr_increment = (sa.scaled_lr - cur_lr) / len(sa.trainloader)
    else:
        sa.adjust_learning_rate(sa.scaled_lr)

    train_loss = 0
    batch_cnt = 0
    num_samples = 0
    now = time.time()
    epoch_start = now

    for step, batch in enumerate(sa.trainloader):
        if warm_up:
            cur_lr += lr_increment
            sa.adjust_learning_rate(cur_lr)

        batch = tuple(t.to(sa.device) for t in batch)
        if sa.name == 'nlp':
            inputs, input_mask, segment_ids, targets = batch
            logits = sa.dist_net(inputs, segment_ids, input_mask)
            loss = sa.criterion(logits.view(-1, sa.num_labels),
                                targets.view(-1))
        elif sa.name == 'googlenet':
            inputs, targets = batch
            outputs, aux1, aux2 = sa.dist_net(inputs)
            loss1 = sa.criterion(outputs, targets)
            loss2 = sa.criterion(aux1, targets)
            loss3 = sa.criterion(aux2, targets)
            loss = loss1 + 0.3 * (loss2 + loss3)
        elif sa.name == 'inception_v3':
            inputs, targets = batch
            outputs, aux_outputs = sa.dist_net(inputs)
            loss1 = sa.criterion(outputs, targets)
            loss2 = sa.criterion(aux_outputs, targets)
            loss = loss1 + 0.4 * loss2
        else:
            inputs, targets = batch
            outputs = sa.dist_net(inputs)
            loss = sa.criterion(outputs, targets)

        sa.optimizer.zero_grad()
        loss.backward()
        sa.optimizer.step()
        train_loss += loss.item()

        num_samples += targets.size(0)
        batch_cnt += 1

        batch_time = time.time() - now
        now = time.time()

        sa.logger.info(
            '[%d/%d] LR: %g | Loss: %.3f | Throughput: %.3f (samples/sec)' %
            (step + 1, len(sa.trainloader), cur_lr, train_loss /
             (step + 1), global_batch_size / (batch_time + 1e-6)))

        if sa.check_pause():
            break

    epoch_end = time.time()
    epoch_loss = train_loss / batch_cnt
    epoch_throughput = num_samples / (epoch_end - epoch_start + 1e-6)
    sa.logger.info('\n[Epoch %d] Loss: %.3f | Throughput: %.3f (samples/sec)' %
                   (epoch, epoch_loss, epoch_throughput))

    return num_samples, epoch_loss, epoch_throughput


def _accuracy(output, target):
    prediction = torch.argmax(output, dim=1)
    return torch.sum(prediction == target).item()


def _eval(sa, epoch):
    sa.dist_net.eval()
    testloader = sa._dist_loader(sa.testset)
    eval_loss = 0
    acc = 0
    total = 0
    num_step = 0
    with torch.no_grad():
        for batch in testloader:
            batch = tuple(t.to(sa.device) for t in batch)
            if sa.name == 'nlp':
                inputs, input_mask, segment_ids, targets = batch
                outputs = sa.dist_net(inputs, segment_ids, input_mask)
                loss = sa.criterion(outputs.view(-1, sa.num_labels),
                                    targets.view(-1))
            else:
                inputs, targets = batch
                outputs = sa.dist_net(inputs)
                loss = sa.criterion(outputs, targets)

            eval_loss += loss.item()
            acc += _accuracy(outputs, targets)
            total += targets.size(0)
            num_step += 1

        eval_loss = eval_loss / num_step
        acc = acc / total
        sa.logger.info('[Epoch %d] Evaluation loss: %.3f | Acc: %.3f%%' %
                       (epoch, eval_loss, acc * 100))
    return eval_loss, acc


def train(sa, num_epochs=None, patience=None):
    assert num_epochs is not None or patience is not None, 'No stopping criterion provided.'
    epoch = sa.start_epoch
    if epoch == 0:
        loss, acc = _eval(sa, epoch)
        epoch = sa.upload_log(epoch, 0, loss=loss, acc=acc)
    while True:
        num_samples, loss, throughput = _train_epoch(sa, epoch)
        if num_samples * sa.size * 2 >= sa.epoch_size:
            loss, acc = _eval(sa, epoch)
            epoch = sa.upload_log(epoch,
                                  num_samples,
                                  throughput=throughput,
                                  loss=loss,
                                  acc=acc)
        else:
            epoch = sa.upload_log(epoch, num_samples, throughput=throughput)
        if sa.check_pause() and sa.flag_scale:
            sa.scale()
        if sa.exit_status in [exit_released, exit_stopped]:
            return epoch
        if num_epochs is not None and epoch > num_epochs:
            break
        if patience is not None and sa.convergence_counter >= patience:
            break
    sa.exit_status = exit_complete
    return epoch


def save(sa, epoch, path):
    if sa.rank == 0:
        state = {
            'net': sa.net.state_dict(),
            'epoch': epoch,
            # 'num_samples': num_samples,
            'lr': sa.get_lr()
        }
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(state, path + 'ckpt.t7')
        sa.logger.info('model saved at %s' % path)
        sa.complete(path)
    else:
        sa.complete()
