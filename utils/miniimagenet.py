from torchvision.transforms import transforms
from MiniImageNet_MAML import MiniImagenet

transform = transforms.Compose([lambda x: x.convert('RGB'),
                                transforms.Resize((84, 84)),
                                # transforms.RandomHorizontalFlip(),
                                # transforms.RandomRotation(5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ])


def trainset(root='./miniimagenet', resize=(84, 84), normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
             transform=None):
    if transform is not None:
        _transform = transform
    else:
        _transform = transforms.Compose([lambda x: x.convert('RGB'),
                                         transforms.Resize(resize),
                                         # transforms.RandomHorizontalFlip(),
                                         # transforms.RandomRotation(5),
                                         transforms.ToTensor(),
                                         transforms.Normalize(*normalize)
                                         ])
    return MiniImagenet(root=root, mode='train', transform=_transform)


def testset(root='./miniimagenet', resize=(84, 84), normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transform=None):
    if transform is not None:
        _transform = transform
    else:
        _transform = transforms.Compose([lambda x: x.convert('RGB'),
                                         transforms.Resize(resize),
                                         # transforms.RandomHorizontalFlip(),
                                         # transforms.RandomRotation(5),
                                         transforms.ToTensor(),
                                         transforms.Normalize(*normalize)
                                         ])
    return MiniImagenet(root=root, mode='test', transform=_transform)
