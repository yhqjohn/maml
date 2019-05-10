import torch
import torch.distributed as dist

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import argparse
from torch.utils import data
import os
import numpy as np

from utils import miniimagenet, to_device
from utils.fewshot import NWaySampler, NWaySlice, relabel
from utils.distributed import DistributedBatchSampler
from meta import Meta
from models import Learner


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def average_model(model):
    """ Parameter averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= size


def run(rank, size, args):
    """ Distributed Synchronous SGD Example """

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    train_set = miniimagenet.trainset()
    batch_sampler = NWaySampler(train_set, args.k_shot+args.k_query, args.n_way)
    batch_sampler = DistributedBatchSampler(train_set, batch_sampler)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=2)

    test_set = miniimagenet.testset()
    test_batch_sampler = NWaySampler(test_set, args.k_shot+args.k_query, args.n_way)
    test_batch_sampler = DistributedBatchSampler(test_set, test_batch_sampler)
    test_loader = DataLoader(test_set, batch_sampler=test_batch_sampler, num_workers=2)

    slc = NWaySlice(args.k_shot, args.k_query, args.n_way)

    net = Learner(config).to(device)
    model = Meta(args, net).to(device)
    average_model(model)
    optimizer = model.meta_optim

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)

    num_batches = ceil(len(train_set) / float(args.batch_size))

    for epoch in range(args.epoch // 500):
        epoch_loss = 0.0
        average_model(model)
        train_iter = zip(*[iter(train_loader)]*args.batch_size)
        for step, data in enumerate(train_iter):
            data = tuple(map(lambda x: slc(to_device(relabel(x), device)), data))
            optimizer.zero_grad()
            if step % 10 == 1:
                with model.logging:
                    loss = model(data)
            else:
                loss = model(data)
            loss.backward()
            if step % 10 == 1:
                accs = torch.FloatTensor(model.log['corrects']).mean(dim=0).tolist()
                print('\rRank ',
                      dist.get_rank(),
                      'step:', step, '\ttraining acc:', accs)
            average_gradients(model)
            optimizer.step()

        if epoch % 5 == 0:  # evaluation
            accs_all_test = []
            for data_test in zip(*[iter(test_loader)]*args.batch_size):
                with model.logging:
                    data_test = tuple(map(lambda x: slc(to_device(relabel(x), device)), data_test))
                    loss = model(data_test)
                    accs_all_test.append(model.log['corrects'])

            # [b, update_step+1]
            accs = torch.FloatTensor(accs_all_test).mean(dim=(0, 1)).tolist()
            print('Rank ',
                  dist.get_rank(), ', epoch ', epoch, ': ',
                  'Test acc:', accs)
            optimizer.zero_grad()


def init_processes(rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_query', type=int, help='k shot for query set', default=15)
    parser.add_argument('--batch_size', type=int, help='meta batch size', default=8)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    parser.add_argument('--imgc', type=int, help='imgc', default=3)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--no_cuda', type=bool, help='use CPU', default=False)
    parser.add_argument('--world_size', type=int, help='world size of parallelism', default=2)
    args = parser.parse_args()
    print(args)
    size = args.world_size
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

