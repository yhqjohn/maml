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

from omniglotNShot import OmniglotNShot


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
    np.random.seed(rank)

    config = [
        ('conv2d', [1, 64, 3], {'stride': 2}),
        ('relu', [True]),
        ('bn2d', [64]),
        ('conv2d', [64, 64, 3], {'stride': 2}),
        ('relu', [True]),
        ('bn2d', [64]),
        ('conv2d', [64, 64, 3], {'stride': 2}),
        ('relu', [True]),
        ('bn2d', [64]),
        ('conv2d', [64, 64, 2]),
        ('relu', [True]),
        ('bn2d', [64]),
        ('flatten',),
        ('linear', [64, 5]),
    ]

    db_train = OmniglotNShot('./omniglot',
                             batchsz=args.task_num//args.world_size,
                             n_way=args.n_way,
                             k_shot=args.k_shot,
                             k_query=args.k_query,
                             imgsz=args.imgsz)

    db_test = OmniglotNShot('./omniglot',
                             batchsz=args.task_num,
                             n_way=args.n_way,
                             k_shot=args.k_shot,
                             k_query=args.k_query,
                             imgsz=args.imgsz)

    # net = CNN(config).to(device)
    net = get_cnn(config)
    net = Learner(net)
    model = Meta(args, net).to(device)
    average_model(model)
    optimizer = model.meta_optim

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)

    # num_batches = ceil(len(train_set) / float(args.batch_size))

    for step in range(args.epoch):
        # data = tuple(map(lambda x: slc(to_device(relabel(x), device)), data))
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        data = list(zip(x_spt, y_spt, x_qry, y_qry))
        optimizer.zero_grad()

        if step * args.task_num % 1600 == 0:
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
        if step * args.task_num % 16000 == 0 and rank == 0:
            accs_all_test = []
            model.eval()
            optimizer.zero_grad()
            for _ in range(1000//args.task_num):
                x_spt, y_spt, x_qry, y_qry = db_test.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
                data = list(zip(x_spt, y_spt, x_qry, y_qry))
                with model.logging:
                    # data_test = tuple(map(lambda x: slc(to_device(relabel(x), device)), data_test))
                    loss = model(data)
                    accs = model.accs()
                    accs_all_test.append(model.log['corrects'])
                    optimizer.zero_grad()

            # [b, update_step+1]
            accs = np.array(accs_all_test).mean(axis=(0, 1)).astype(np.float16)
            print('Rank ',
                  dist.get_rank(), ', step ', step, ': ',
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
    parser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    parser.add_argument('--n_way', type=int, help='n way', default=5)
    parser.add_argument('--k_shot', type=int, help='k shot for support set', default=1)
    parser.add_argument('--k_query', type=int, help='k shot for query set', default=15)
    parser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    parser.add_argument('--imgc', type=int, help='imgc', default=1)
    parser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
    parser.add_argument('--no_cuda', type=bool, help='use CPU', default=False)
    parser.add_argument('--world_size', type=int, help='world size of parallelism', default=2)
    parser.add_argument('--addr', type=str, help='master address', default='127.0.0.1')
    parser.add_argument('--port', type=str, help='master port', default='29500')
    args = parser.parse_args()
    print(args)
    size = args.world_size
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run, args), kwargs={'addr': args.addr, 'port': args.port})
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

