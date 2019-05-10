from torch.utils.data import Dataset, Sampler, DataLoader, BatchSampler
from torchvision.transforms import transforms
import numpy as np
from PIL import Image
import pandas, os
from utils import labelling, make_n_way_dict


class MiniImagenet(Dataset):
    def __init__(self, root, mode='train', transform=None, transform_target=None):
        super(MiniImagenet, self).__init__()

        # get data from file
        if mode in ('train', 'val', 'test'):
            data_info = pandas.read_csv(os.path.join(root, mode+'.csv'))
        else:
            raise ValueError('Only train, val and test mode are supported')
        data = np.asarray(data_info)
        labelling(data[:, 1])
        self.data = np.asarray(sorted(data, key=lambda a_entry: a_entry[1])) # sort data according to label

        # set the preprocessing method for the data
        self.img_root = os.path.join(root, 'images/')
        self.transform = lambda x: x if transform is None else transform(x)
        self.img_process = lambda x: self.transform(Image.open(
                os.path.join(
                    self.img_root,
                    x
                )
            ))
        self.transform_target = lambda x: x if transform_target is None else transform_target(x)

        # making image to label dict and label to index list dict
        self.idx2label, self.label2idx = make_n_way_dict(self.data)

        # count length
        self.data_len = len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.img_process(img), self.transform_target(label)

    def __len__(self):
        return self.data_len








