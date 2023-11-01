"""
_summary_
    I want to compare each models
        - specificity
"""
import os
from abc import ABC
from dataclasses import dataclass

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import pandas as pd
from thop import profile, clever_format
from torch.utils.data import DataLoader


@dataclass
class ModelComparator(ABC):
    """
    _summary_

    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        args (_type_): _description_
        device (_type_): _description_
    """
    models: list
    model_names: list
    test_loader: DataLoader
    g1_identifiers: list
    device: torch.device

    def perform_tests(self, filepath):
        """
        _summary_

        Returns:
            _type_: _description_
        """
        results = []
        for model in self.models:
            model.eval()

        print("Testing for model FLOPS", flush=True)
        macs, params = self._average_flops()

        print("Testing average memory size", flush=True)
        memory_sizes = self._memory_size()

        print("Testing for Top 1 Accuracies", flush=True)
        top1_accuracies = self._topk_accuracy(k=1)

        print("Testing for Top 5 Accuracies", flush=True)
        top5_accuracies = self._topk_accuracy(k=5)

        print("Making Confusion Matrices")
        self._get_confusion_matrix(save_path='../results/confusion_matrix')

        print("Structuring Results", flush=True)
        for i, (model_name, _) in enumerate(zip(self.model_names, self.models)):
            results.append({
                "Model": f"{model_name} Model",
                "Top-1 Accuracy (%)": top1_accuracies[i],
                "Top-5 Accuracy (%)": top5_accuracies[i],
                "Average MACs": macs[i],
                "Number Params": params[i],
                "Memory Size (MB)": memory_sizes[i]
            })
        model_results = pd.DataFrame(results)
        model_results.to_csv(f'{filepath}/comparisons.csv')

    def _topk_accuracy(self, k=1):
        """
        _summary_

        Args:
            k (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        accuracies = []
        for model in self.models:
            model.to(self.device)
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # get new labels
                    new_labels = [self.g1_identifiers[i.item()] for i in labels]
                    new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

                    outputs, exits = model(inputs)
                    print(exits)
                    _, predicted = outputs.topk(k, 1, largest=True, sorted=True)

                    if exits and 'coarse' in exits:
                        labels = new_labels

                    total += labels.size(0)
                    correct += (predicted == labels.view(-1, 1)).sum().item()

            accuracy = 100 * correct / total
            accuracies.append(accuracy)
        return accuracies

    def _get_confusion_matrix(self, save_path):
        """
        _summary_

        Args:
            save_path (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        for model, model_name in zip(self.models, self.model_names):
            model.to(self.device)
            model.eval()
            y_true = []
            y_pred = []

            label_names = sorted(set(self.g1_identifiers))

            with torch.no_grad():
                for data in self.test_loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # get new labels
                    new_labels = [self.g1_identifiers[i.item()] for i in labels]
                    new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

                    outputs, exits = model(inputs)
                    _, predicted = outputs.topk(1, 1, largest=True, sorted=True)

                    if exits and 'coarse' in exits:
                        labels = new_labels

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.view(-1).cpu().numpy())

            confusion_matrix_data = confusion_matrix(y_true, y_pred)

            # Create and save the confusion matrix plot
            plt.figure(figsize=(16, 13))
            ax = sns.heatmap(confusion_matrix_data, annot=False, cmap='rocket', cbar=True)

            # Set custom labels on x and y axes
            ax.set_xticks(np.arange(len(label_names)) + 0.5)
            ax.set_yticks(np.arange(len(label_names)) + 0.5)
            ax.set_xticklabels(label_names, rotation=270, fontsize=10)
            ax.set_yticklabels(label_names, rotation=0, fontsize=10)

            plt.xlabel('Predicted', fontsize=14)
            plt.ylabel('Actual', fontsize=14)
            plt.title(f'Confusion Matrix ({model_name} Model)', fontsize=18)
            plt.savefig(f'{save_path}/{model_name}.png')
            plt.close()

    def _average_flops(self):
        """
        _summary_

        Returns:
            _type_: _description_
        """
        macs, params = [], []
        for model in self.models:
            model.to(self.device)
            x = torch.randn(1, 3, 32, 32).to(self.device)
            mac, params = profile(model, inputs=(x, ))
            mac, params = clever_format([mac, params], "%.3f")
            macs.append(mac)
            print("\n")

        return macs, params

    def _memory_size(self):
        """
        _summary_

        Returns:
            _type_: _description_
        """
        sizes = []
        for model in self.models:
            torch.save(model, 'temp.pth')
            size = os.path.getsize('temp.pth') / (1024.0 ** 2)  # in megabytes
            sizes.append(size)
            os.remove('temp.pth')
        return sizes
