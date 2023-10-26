"""
_summary_
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# import dataset
def dataset():
    """
    _summary_

    Returns:
        _type_: _description_
    """
    # define the transforms
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([transforms.RandomCrop(32, padding=4,
                                          padding_mode='reflect'),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(*stats, inplace=True)])

    # get CIFAR100 data
    train_data = torchvision.datasets.CIFAR100(root='../cifar100/data',
                                               download=False, train=True,
                                               transform=transform)
    test_data = torchvision.datasets.CIFAR100(root='../cifar100/data',
                                              download=False, train=False,
                                              transform=transform)

    return train_data, test_data

# split data into [train, val, test]
def split_data(train_data):
    """
    _summary_

    Args:
        train_data (_type_): _description_
        test_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Split training data into training and validation data
    train_data, val_data = torch.utils.data.random_split(train_data, [45000, 5000])

    return train_data, val_data

# get the fine granularity classes
def g0_classes(train_data):
    """
    _summary_

    Args:
        train_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_classes = len(train_data.classes)
    g0 = [i for i in range(len(train_data.classes))]
    return g0, train_data.classes, num_classes

# get the coarse granularity classes
def g1_classes():
    """
    _summary_

    Returns:
        _type_: _description_
    """
    coarse_labels = ['aquatic mammals', 'fish', 'flowers', 'food containers',
                     'fruit and vegetables', 'household electrical devices', 'household furniture', 
                     'insects', 'large carnivores', 'large man-made outdoor things', 
                     'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 
                     'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

    coarse_classes = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                      3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                      6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                      0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                      5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                      16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                      10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                      2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                      16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                      18, 1, 2, 15, 6, 0, 17, 8, 14, 13]

    return coarse_classes, coarse_labels, len(coarse_labels)

# define dataloader
def dataloader(train_data, val_data, test_data, batch_size):
    """
    _summary_

    Args:
        train_data (_type_): _description_
        val_data (_type_): _description_
        test_data (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    # set up DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,
                              num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader
