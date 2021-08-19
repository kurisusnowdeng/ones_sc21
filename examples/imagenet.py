import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)
from src import scaling
from src.training import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epoch_size', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument(
        '--model',
        type=str,
        help='alexnet/resnet50/vgg16/vgg19/googlenet/inception_v3')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--target', type=float)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--manager_addr', type=str, default='localhost')
    parser.add_argument('--manager_port', type=int, default=17834)
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--port', type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


ARGS = get_args()

device = 'cuda'
torch.backends.cudnn.benchmark = True

sa = scaling.ScalingAgent(ARGS.job_id, ARGS.model, ARGS.local_rank, ARGS.port,
                          ARGS.manager_addr, ARGS.manager_port)

# Data
print('==> Preparing data..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(299 if ARGS.model == 'inception_v3' else 224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_path = os.path.join(ARGS.data_dir, 'train')
trainset = torchvision.datasets.ImageFolder(train_path,
                                            transform=transform_train)
trainset = torch.utils.data.Subset(trainset, range(ARGS.epoch_size))

test_path = os.path.join(ARGS.data_dir, 'val')
testset = torchvision.datasets.ImageFolder(test_path, transform=transform_test)
k = ARGS.epoch_size // 1000
if ARGS.epoch_size % 1000 > 0:
    k += 1
testset = torch.utils.data.Subset(testset, range(k * 50))

print('==> Loaded %d training samples, %d test samples' %
      (len(trainset), len(testset)))

# Model
print('==> Building model: ' + ARGS.model)
net = models.__dict__[ARGS.model]()

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if ARGS.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(
        ARGS.model_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(ARGS.model_dir + '/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=ARGS.lr,
                      momentum=0.9,
                      weight_decay=1e-4)

sa.load(net=net,
        criterion=criterion,
        optimizer=optimizer,
        trainset=trainset,
        testset=testset,
        batch_size=ARGS.batch_size,
        lr=ARGS.lr,
        num_labels=1000,
        start_epoch=start_epoch,
        scale=True if ARGS.scale else False)

if ARGS.resume:
    sa.adjust_learning_rate(checkpoint['lr'])

epoch = train(sa, patience=ARGS.patience)
save(sa, epoch, ARGS.model_dir)
