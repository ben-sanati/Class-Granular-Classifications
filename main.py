"""
_summary_
"""
import warnings
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchinfo import summary

# models
from models.alexnet import AlexNet
from models.b_alexnet import BranchyAlexNet
from processing.training import AlexNetTrainer, BranchyNetTrainer
from processing.testing import ModelComparator
from utils.load_data import dataset, split_data, g0_classes, g1_classes, dataloader

def get_data(batch_size: int):
    """
    _summary_

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get data
    train_data, test_data = dataset()

    # get classes info
    g0, g0_names, num_g0_classes = g0_classes(train_data=train_data)
    g1, g1_names, num_g1_classes = g1_classes()

    # split data into [train, val, test]
    train_data, val_data = split_data(train_data=train_data)

    # put data into dataloader
    train, val, test = dataloader(train_data=train_data, val_data=val_data,
                                                       test_data=test_data,
                                                       batch_size=batch_size)

    return train, val, test

def training(_data: tuple, _models: dict, _trainer: dict,
             _args: argparse.Namespace, _model_path: str):
    """
    _summary_

    Args:
        _data (tuple): _description_
        _models (dict): _description_
        _trainer (dict): _description_
        _args (argparse.Namespace): _description_
        _model_path (str): _description_
    """
    train_loader, val_loader, test_loader = _data
    for model_name, model_def in _models.items():
        # initialize the model and summarise it
        device = torch.device(_args.device)
        model = model_def().to(device)
        print(f"Model == {model_name}\n")

        _temp_train, _, _ = _data
        images, _ = next(iter(_temp_train))
        summary(model, input_size=[images.shape], dtypes=[torch.float32])

        # definitions
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr,
                                     weight_decay=_args.weight_decay)

        # train and plot results
        trainer = _trainer[model_name](model, train_loader, val_loader, test_loader,
                                       loss_fn, optimizer, _args, device)
        trainer.train(filepath=f'{_model_path}/{model_name}.pth')
        trainer.plot_and_save(model_name=model_name,
                              save_folder=f'../results/training_plots/{model_name}')
        print("\n" + "||" + "="*40 + "||" + "\n")

def testing(_data: tuple, _models: dict, _model_paths: dict,
            _args: argparse.Namespace, folder: str):
    """
    _summary_

    Args:
        _data (tuple): _description_
        _models (dict): _description_
        _trainer (dict): _description_
        _args (argparse.Namespace): _description_
        _model_path (str): _description_
    """
    print("Testing Mode\n" + "="*50)

    # load models
    device = torch.device(_args.device)
    train_loader, val_loader, test_loader = _data
    models, model_names = [], []
    for model_name, model_path in _model_paths.items():
        model = _models[model_name]().to(device)
        model.load_state_dict(torch.load(model_path))
        model_names.append(model_name)
        models.append(model)

    model_tester = ModelComparator(models, model_names, test_loader, device)
    model_tester.perform_tests(folder)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # make the output deterministic
    SEED = 7
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # choice definitions
    MODEL_PATH = '../results/models'
    COMPARATOR_PATH = '../results/model_comparators'
    MODE_MAP = {'training': training, 'testing': testing}
    MODELS = {'AlexNet': AlexNet, 'Branchy-AlexNet': BranchyAlexNet}  # {'AlexNet': AlexNet, 'Branchy-AlexNet': BranchyAlexNet, 'Super-HBN': Super_HBN}
    TRAINER = {'AlexNet': AlexNetTrainer, 'Branchy-AlexNet': BranchyNetTrainer}  # {'AlexNet': AlexNetTrainer, 'Branchy-AlexNet': BranchyNetTrainer, 'Super-HBN': super_train}
    MODEL_PATHS = {'AlexNet': f'{MODEL_PATH}/AlexNet.pth', 'Branchy-AlexNet': f'{MODEL_PATH}/Branchy-AlexNet.pth'}

    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODE_MAP.keys(), type=str, default='training')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_epochs', type=int, default=2)
    args = parser.parse_args()

    # print run details
    print("Args:")
    for key, value in vars(args).items():
        print(f"\t{key}: {value}")
    print("\n")

    # get data
    data = get_data(args.batch_size)

    # train/test models
    if args.mode == 'training':
        MODE_MAP[args.mode](data, MODELS, TRAINER, args, MODEL_PATH)
    else:
        MODE_MAP[args.mode](data, MODELS, MODEL_PATHS, args, COMPARATOR_PATH)
