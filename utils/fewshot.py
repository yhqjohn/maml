from torch.utils.data import Sampler, DataLoader
import random
import numpy as np
from utils import make_n_way_dict, grouper, labelling


class NWaySampler(Sampler):
    def __init__(self, datasource, batch_size, n_way, seed=None):

        # get label2idx and idx2label
        if hasattr(datasource, 'label2idx') and hasattr(datasource, 'idx2label'):
            self.label2idx, self.idx2label = datasource.label2idx, datasource.idx2label
        else:
            self.label2idx, self.idx2label = make_n_way_dict(datasource)

        self.batch_size = batch_size
        self.n_way = n_way
        self.generator = np.random.RandomState(seed=seed)

    def set_seed(self, seed):
        self.generator.seed(seed)

    def _generate_sample_list(self, batchsize):
        def batch_sampler(label2idx):
            n_way_idx_list = [[j for j in label2idx[i]] for i in label2idx]
            for i in n_way_idx_list:
                self.generator.shuffle(i)

            p_arr = np.array([len(i) for i in n_way_idx_list])
            p_arr[p_arr <= batchsize] = 0
            while p_arr.any():
                p_arr = (lambda x: x/x.sum())(p_arr)    # normalize to sum 1
                class_id = self.generator.choice(len(n_way_idx_list), p=p_arr)
                yield [n_way_idx_list[class_id].pop(-1) for i in range(batchsize)]
                p_arr = np.array([len(i) for i in n_way_idx_list])
                p_arr[p_arr <= batchsize] = 0

        sample_list = np.asarray([i for i in batch_sampler(self.label2idx)])
        sample_list = sample_list[:len(sample_list)//self.n_way*self.n_way]
        sample_list = sample_list.reshape(-1, self.n_way, batchsize)

        return [i.T.flatten().tolist() for i in sample_list]

    def __iter__(self):
        sample_list = self._generate_sample_list(self.batch_size)
        assert len(sample_list) == len(self)
        return iter(sample_list)

    def __len__(self):
        l1 = 0
        for i in self.label2idx:
            l1 += len(self.label2idx[i])//self.batch_size

        return l1//self.n_way


def relabel(x):
    labelling(x[1], shuffle=True)
    return x


class NWaySlice:
    def __init__(self, k_shot, k_query, n_way=None):
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        if isinstance(n_way, int):
            self.support_size = n_way*k_shot
            self.query_size = n_way*k_query
        elif n_way is None:
            self.support_size = k_shot
            self.query_size = k_query

    def __call__(self, x):
        data, labels = x
        support_x, query_x = data[:self.support_size], data[self.support_size:]
        support_y, query_y = labels[:self.support_size], labels[self.support_size:]
        return support_x, support_y, query_x, query_y
