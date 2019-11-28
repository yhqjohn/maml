import torch
import torch.distributed as dist

from torch.multiprocessing import Process
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate as collate

import argparse
import os
import numpy as np

from maml import Meta
from models import get_cnn
from utils.data import task_loader
from metann import Learner
import learn2learn as l2l


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

    device = torch.device(args.device)

    config = [
        ('conv2d', [3, 32, 3]),
        ('relu', [True]),
        ('bn2d', [32]),
        ('max_pool2d', [2, 2]),
        ('conv2d', [32, 32, 3]),
        ('relu', [True]),
        ('bn2d', [32]),
        ('max_pool2d', [2, 2]),
        ('conv2d', [32, 32, 3]),
        ('relu', [True]),
        ('bn2d', [32]),
        ('max_pool2d', [2, 2]),
        ('conv2d', [32, 32, 3]),
        ('relu', [True]),
        ('bn2d', [32]),
        ('max_pool2d', [2, 1]),
        ('flatten',),
        ('linear', [32 * 5 * 5, 5]),
    ]


    train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train')
    # valid_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='validation')
    test_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='test')
    # train_loader = task_loader(train_dataset, args.n_way, args.k_shot, args.k_query, 10000,
    #                            batch_size=args.task_num//args.world_size)
    # test_loader = task_loader(test_dataset, args.n_way, args.k_shot, args.k_query, 1024,
    #                            batch_size=args.task_num//args.world_size)

    net = get_cnn(config) #要改
    model = Meta(update_lr=args.update_lr, meta_lr=args.meta_lr, update_step=args.update_step,
                 update_step_test=args.update_step_test, learner=Learner(net)).to(device)
    average_model(model)
    optimizer = model.meta_optim

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)

    # num_batches = ceil(len(train_set) / float(args.batch_size))

    # for epoch in range(args.epoch):
    for epoch in range(args.epoch // 10000):
        epoch_loss = 0.0
        average_model(model)
        train_loader = task_loader(train_dataset, args.n_way, args.k_shot, args.k_query, 10000,
                                   batch_size=args.task_num // args.world_size)
        for step, data in enumerate(train_loader):
            # data = tuple(map(lambda x: slc(to_device(relabel(x), device)), data))
            data = [[x.to(device) for x in collate(a) + collate(b)] for a, b in data]
            optimizer.zero_grad()
            if step * args.task_num % 120 == 0:
                with model.logging:
                    loss = model(data)
                accs = model.accs()
                print('\rRank ',
                      dist.get_rank(),
                      'step:', step, '\ttraining acc:', accs)
            else:
                loss = model(data)
            loss.backward()
            average_gradients(model)
            optimizer.step()

            # if epoch % 5 == 0:  # evaluation
            if step * args.task_num % 2000 == 0:
                accs_all_test = []
                test_loader = task_loader(test_dataset, args.n_way, args.k_shot, args.k_query, 1024,
                                          batch_size=args.task_num // args.world_size)
                model.eval()
                for data_test in test_loader:
                    data_test = [[x.to(device) for x in collate(a) + collate(b)] for a, b in data_test]
                    with model.logging:
                        # data_test = tuple(map(lambda x: slc(to_device(relabel(x), device)), data_test))
                        loss = model(data_test)
                        loss.backward()
                        # accs = model.accs()
                        accs_all_test.append(model.log['corrects'])
                        optimizer.zero_grad()

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Rank ',
                      dist.get_rank(), ', epoch ', epoch, ': ',
                      'Test acc:', accs)
                optimizer.zero_grad()
                del data_test
                model.train()


def init_processes(rank, size, fn, args, backend='gloo', addr='127.0.0.1', port='29500'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_query', type=int, help='k shot for query set', default=15)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--device', type=str, help='use CPU', default='cuda')
    parser.add_argument('--world_size', type=int, help='world size of parallelism', default=1)
    parser.add_argument('--rank', type=int, help='rank', default=0)
    parser.add_argument('--addr', type=str, help='master address', default='127.0.0.1')
    parser.add_argument('--port', type=str, help='master port', default='29500')
    args = parser.parse_args()
    print(args)
    size = args.world_size
    rank = args.rank
    init_processes(rank, size, run, args, addr=args.addr, port=args.port)

