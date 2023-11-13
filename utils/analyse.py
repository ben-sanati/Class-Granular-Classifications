import os
from time import perf_counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
from thop import profile, clever_format

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.alexnet import *
from models.b_alexnet import *
from models.sem_hbn import *
from models.super_hbn import *
from models.td_hbn import *

@dataclass
class Analyse(ABC):

    model_class: nn.Module
    filepath: str
    device: torch.device
    coarse_converter: list
    threshold: list
    fine_tolerance: list
    test_loader: DataLoader
    model: nn.Module = None
    branch1_exits: int = 0
    branch2_exits: int = 0
    branch3_exits: int = 0
    fine_exits: int = 0
    coarse_exits: int = 0

    def perform_analysis(self):
        num_correct = 0
        num_samples = 0
        hierarchical_correct = 0
        hierarchical_samples = 0
        specificities = []

        # Resetting the attributes to zero at the beginning of each loop
        self.branch1_exits = 0
        self.branch2_exits = 0
        self.branch3_exits = 0
        self.fine_exits = 0
        self.coarse_exits = 0

        # load model
        self.model = self.model_class().to(self.device)
        self.model.load_state_dict(torch.load(self.filepath, map_location=self.device))
        self.model.eval()

        with torch.no_grad():
            print("Timer Active")
            print(f"# Testing Iterations: {len(self.test_loader)}")
            tic = perf_counter()

            imagi = []
            for index, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                imagi.append(images)

                # get coarse label conversions from fine labels
                new_labels = [self.coarse_converter[i.item()] for i in labels]
                new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

                # get predictions
                outputs, exit = self._get_output(images, labels)

                # get accuracy
                _, prediction = torch.max(outputs, dim=1)
                num_samples += labels.shape[0]

                if exit and "coarse" in exit:
                    num_correct += (prediction == new_labels).sum().item()
                    hierarchical_prediction = prediction

                    self.coarse_exits += 1
                    specificities.append(0)
                else:
                    num_correct += (prediction == labels).sum().item()

                    # top 1 convert fine prediction to coarse label
                    hierarchical_prediction = torch.tensor([self.coarse_converter[predict.item()] for predict in prediction]).to(self.device)

                    self.fine_exits += 1
                    specificities.append(1)

                hierarchical_samples += new_labels.shape[0]
                hierarchical_correct += (hierarchical_prediction == new_labels).sum().item()

                # get number of branch exits
                if exit and '1' in exit:
                    self.branch1_exits += 1
                elif exit and '2' in exit:
                    self.branch2_exits += 1
                elif exit and '3' in exit:
                    self.branch3_exits += 1

            toc = perf_counter()
            print("Timer Ended")

            # get the FLOPs
            print("Getting FLOPs")
            x = torch.randn(1, 3, 32, 32).to(self.device)
            flops = self._get_flops(x)

            # get the memory size
            print("Getting memory size")
            torch.save(self.model, 'temp.pth')
            memory = os.path.getsize('temp.pth') / (1024.0 ** 2)  # in megabytes
            os.remove('temp.pth')

        state_dict = self.model.state_dict()
        state_dict = {key: value.cpu().numpy() for key, value in state_dict.items()}

        test_accuracy = 100 * num_correct / num_samples
        hierarchical_accuracy = 100 * hierarchical_correct / hierarchical_samples
        specificity = sum(specificities) / len(specificities)
        time_taken = toc - tic

        return test_accuracy, hierarchical_accuracy, specificity, flops, memory, time_taken, self.branch1_exits, self.branch2_exits, self.branch3_exits, self.fine_exits, self.coarse_exits

    @abstractmethod
    def _get_output(self, images, labels):
        """
        """

    @abstractmethod
    def _get_flops(self, x):
        """
        """


@dataclass
class AnalyseAlexNet(Analyse):
    def _get_output(self, images, labels):
        with torch.no_grad():
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _get_flops(self, x):
        macs, params = profile(self.model, inputs=(x, ))
        mac, params = clever_format([macs, params], "%.3f")
        flops = 2 * macs
        return flops

@dataclass
class AnalyseBranchyNet(Analyse):
    def _get_output(self, images, labels):
        with torch.no_grad():
            y_hat, exits = self.model(images, threshold=self.threshold)
            return y_hat, exits

    def _get_flops(self, x):
        macs, params = profile(self.model, inputs=(x, self.threshold, ))
        mac, params = clever_format([macs, params], "%.3f")
        flops = 2 * macs
        return flops

@dataclass
class AnalyseSemHBN(Analyse):
    def _get_output(self, images, labels):
        with torch.no_grad():
            y_hat, exits = self.model(images, threshold=self.threshold)
            return y_hat, exits

    def _get_flops(self, x):
        macs, params = profile(self.model, inputs=(x, self.threshold, ))
        mac, params = clever_format([macs, params], "%.3f")
        flops = 2 * macs
        return flops

@dataclass
class AnalyseHBN(Analyse):
    def _get_output(self, images, labels):
        with torch.no_grad():
            y_hat, exits = self.model(images, threshold=self.threshold, fine_tolerance=self.fine_tolerance)
            return y_hat, exits

    def _get_flops(self, x):
        macs, params = profile(self.model, inputs=(x, self.threshold, self.fine_tolerance, ))
        mac, params = clever_format([macs, params], "%.3f")
        flops = 2 * macs
        return flops

@dataclass
class AnalyseTDHBN(Analyse):
    def _get_output(self, images, labels):
        with torch.no_grad():
            y_hat, exits = self.model(images, threshold=self.threshold)
            return y_hat, exits

    def _get_flops(self, x):
        macs, params = profile(self.model, inputs=(x, self.threshold, ))
        mac, params = clever_format([macs, params], "%.3f")
        flops = 2 * macs
        return flops
