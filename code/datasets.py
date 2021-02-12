import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from torch.utils.data.sampler import SubsetRandomSampler

####################################################################
######################       CIFAR           #######################
####################################################################


def get_cifar_train_valid_loader(batch_size,
                               augment,
                               random_seed,
                               valid_size=0.1,
                               shuffle=True,
                               num_workers=1,
                               pin_memory=False,
                               dataset_name='CIFAR10',
                               centre_and_scale = True):
    """
    Returns:

    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    if dataset_name=='CIFAR10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    elif dataset_name=='CIFAR100':
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )

    # define transforms

    if augment:
        if centre_and_scale:
            print("******* Augment, Centre and scale ********")
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        else:
            print("******* Augment, No centre and scale ********")
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            valid_transform = transforms.Compose([
                transforms.ToTensor()
            ])

    else:
        if centre_and_scale:
            print("******* No Augment, Centre and scale ********")
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

            valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        else:
            print("******* No Augment, No centre and scale ********")
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            valid_transform = transforms.Compose([
                transforms.ToTensor()
            ])
    # load the dataset
    data_dir = '../data'

    if dataset_name == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )
    elif dataset_name == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_cifar_test_loader(batch_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False,
                    dataset_name='CIFAR10',
                    centre_and_scale = True):
    """
    Returns:

    - data_loader: test set iterator.
    """
    if dataset_name=='CIFAR10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
    elif dataset_name=='CIFAR100':
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )

    # define transform
    if centre_and_scale:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # load the dataset
    data_dir = '../data'
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )


    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

####################################################################
########################## Tiny Imagenet ###########################
####################################################################

def get_tiny_imagenet_train_valid_loader(batch_size,
                                           augment,
                                           shuffle=True,
                                           num_workers=1):
    """
    Needs to have Tiny Imagenet already downloaded in ../data/.
    Run 'get_tiny_imagenet.sh' from inside ./data/ before running these experiments
    Returns train, test and val dataloaders - but since there are no test labels for Tiny Imagenet's test set,
    both test and val dataloaders use validation data.
    """

    root = '../data'
    tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
    tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
    transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])


    trainset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/train',
                                            transform=transform_train)
    valset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val',
                                           transform=transform_test)
    testset = torchvision.datasets.ImageFolder(root + '/tiny-imagenet-200/val',
                                           transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    return trainloader, valloader, testloader
