import torch
import torch.distributed as dist

from torch.multiprocessing import Process
from torch.utils.data import DataLoader, DistributedSampler

import argparse
import os
import numpy as np

from maml import Meta
from models import get_cnn
from metann import Learner

from MiniImagenet import MiniImagenet


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


    mini = MiniImagenet('./miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_shot,
                        k_query=args.k_query,
                        batchsz=10000, resize=args.imgsz) #要改
    mini_test = MiniImagenet('./miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_shot,
                             k_query=args.k_query,
                             batchsz=100, resize=args.imgsz) #要改

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
        train_iter = DataLoader(mini, args.task_num//args.world_size, num_workers=args.world_size, sampler=DistributedSampler(mini), pin_memory=True)
        for step, data in enumerate(train_iter):
            # data = tuple(map(lambda x: slc(to_device(relabel(x), device)), data))
            data = [i.to(device) for i in data]
            data = list(zip(*data))
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
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                model.eval()
                optimizer.zero_grad()
                for data_test in db_test:
                    data_test = [i.to(device) for i in data_test]
                    data_test = list(zip(*data_test))
                    with model.logging:
                        # data_test = tuple(map(lambda x: slc(to_device(relabel(x), device)), data_test))
                        loss = model(data_test)
                        accs = model.accs()
                        accs_all_test.append(model.log['corrects'])
                        optimizer.zero_grad()

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Rank ',
                      dist.get_rank(), ', epoch ', epoch, ': ',
                      'Test acc:', accs)
                optimizer.zero_grad()
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
    parser.add_argument('--batch_size', type=int, help='meta batch size', default=2)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    parser.add_argument('--imgc', type=int, help='imgc', default=3)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument('--device', type=str, help='use CPU', default='cuda')
    parser.add_argument('--world_size', type=int, help='world size of parallelism', default=2)
    parser.add_argument('--rank', type=int, help='rank', default=0)
    parser.add_argument('--addr', type=str, help='master address', default='127.0.0.1')
    parser.add_argument('--port', type=str, help='master port', default='29500')
    args = parser.parse_args()
    print(args)
    size = args.world_size
    rank = args.rank
    init_processes(rank, size, run, args, addr=args.addr, port=args.port)

