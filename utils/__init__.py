import torch
import numpy as np


# convert a list of n kinds of objects into a list of labels from 0 to n-1
def labelling(l, startidx=0, shuffle=False, seed=None):
    def data(a):
        try:
            return a.item()
        except (NameError, ValueError, AttributeError):
            return a
    d = {}
    idx = startidx
    for k, v in enumerate(l):
        if data(v) in d:
            l[k] = d[data(v)]
        else:
            d[data(v)] = idx
            idx += 1
            l[k] = d[data(v)]

    if shuffle:
        generator = np.random.RandomState(seed=seed)
        labels = [i for i in range(startidx, idx)]
        generator.shuffle(labels)
        for k, v in enumerate(l):
            l[k] = labels[l[k]]


def make_n_way_dict(data):
    idx2label = [0]*len(data)
    label2idx = {}
    for k, v in enumerate(data):
        idx2label[k] = v[1]
        if v[1] in label2idx:
            label2idx[v[1]].add(k)
        else:
            label2idx[v[1]] = {k, }
    return idx2label, label2idx


def grouper(iterable, n):
    _iter = iter(iterable)
    _iter = [_iter]*n
    try:
        while True:
            values = []
            for it in _iter:
                values.append(next(it))
            yield values
    except StopIteration:
        return


def to_device(a, device):
    if isinstance(a, torch.Tensor):
        return a.to(device)
    elif not a:
        return ()
    else:
        return (to_device(a[0], device),) + to_device(a[1:], device)
