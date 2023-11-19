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

        self._temp_func()

        # print("Testing for model FLOPS", flush=True)
        # macs, params = self._average_flops()

        # print("Testing average memory size", flush=True)
        # memory_sizes = self._memory_size()

        # print("Testing for Top 1 Accuracies", flush=True)
        # top1_accuracies = self._topk_accuracy(k=1)

        # print("Testing for Top 5 Accuracies", flush=True)
        # top5_accuracies = self._topk_accuracy(k=5)

        # print("Making Confusion Matrices")
        # self._get_confusion_matrix(save_path='../results/confusion_matrix')

        # print("Structuring Results", flush=True)
        # for i, (model_name, _) in enumerate(zip(self.model_names, self.models)):
        #     results.append({
        #         "Model": f"{model_name} Model",
        #         "Top-1 Accuracy (%)": top1_accuracies[i],
        #         "Top-5 Accuracy (%)": top5_accuracies[i],
        #         "Average MACs": macs[i],
        #         "Number Params": params[i],
        #         "Memory Size (MB)": memory_sizes[i]
        #     })
        # model_results = pd.DataFrame(results)
        # model_results.to_csv(f'{filepath}/comparisons.csv')

    def _temp_func(self):
        epoch_values = []
        specificity_values = []
        top1_acc_values = []

        with open('./training.out', 'r') as file:
            lines = file.readlines()

            i = 0
            current_epoch = None
            processing = False  # flag to control processing

            while i < len(lines):
                line = lines[i]

                if 'Post-Training' in line:
                    processing = True  # set flag to start processing

                if processing:
                    if 'Epoch' in line:
                        current_epoch = int(line.split('[')[1].split('/')[0])
                        epoch_values.append(current_epoch)

                    if 'Specificity' in line:
                        specificity = float(line.split(':')[-1].strip())
                        specificity_values.append(specificity)

                    if 'Top 1 Acc' in line:
                        top1_acc = float(line.split('=')[-1].replace('%', '').strip())
                        top1_acc_values.append(top1_acc)

                i += 1

        # plot 1
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Top 1 Accuracy (%)')
        ax1.plot(epoch_values, top1_acc_values, c='r', label='Top 1 Accuracy')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Specificity')
        ax2.plot(epoch_values, specificity_values, c='b', label='Specificity')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.title('Accuracy and Specificity over Epochs\nduring Post-Training')

        plt.savefig('../results/accuracy-specificity/fine_weighting=1.5.png')

        # plot 2
        fig1, ax = plt.subplots()

        ax.plot(epoch_values, [x * y for x, y in zip(top1_acc_values, specificity_values)])
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy * Specificity')
        plt.title('Accuracy * Specificity over Epochs\nduring Post-Training')

        plt.savefig('../results/accuracy-specificity/fine_weighting_summary=1.5.png')

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
