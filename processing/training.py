"""
_summary_
"""
import argparse
from typing import List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class Trainer(ABC):
    """
    _summary_

    Args:
        model (_type_): _description_
        data (_type_): _description_
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
        args (_type_): _description_
        device (_type_): _description_
        epochs (_type_): _description_
        losses (_type_): _description_
        val1_acc (_type_): _description_
        val5_acc (_type_): _description_
    """
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    loss_fn: nn.Module
    optimizer: torch.optim.Optimizer
    args: argparse.Namespace
    device: torch.device
    epochs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    val1_acc: List[float] = field(default_factory=list)
    val5_acc: List[float] = field(default_factory=list)

    def train(self, filepath: str):
        """
        _summary_
        """
        num_iterations = len(self.train_loader)

        print(f"\n# Training iterations per epoch : {num_iterations}\n")
        print("-"*30 + "\n|" + " "*10 + "Training" + " "*10 + "|\n" + "-"*30 + "\n")
        for epoch in range(self.args.num_epochs):
            for index, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                loss = self._get_output(images, labels)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (index + 1) % num_iterations == 0:
                    top1_val, top5_val = self.validate()
                    print(f"Epoch [{epoch + 1}/{self.args.num_epochs}]: \
                            \n\t\tTop 1 Acc = {top1_val}%\n\t\tTop 5 Acc  = {top5_val}%\n")

                    # add data to lists
                    self.epochs.append(epoch+1)
                    self.losses.append(loss.item())
                    self.val1_acc.append(top1_val)
                    self.val5_acc.append(top5_val)

        print("-" * 30)

        # save model
        torch.save(self.model.state_dict(), filepath)

    @abstractmethod
    def _get_output(self, images, labels):
        """
        _summary_
        """

    def validate(self):
        """
        _summary_
        """
        self.model.eval()
        with torch.no_grad():
            # validation accuracy
            n_correct = 0
            n_samples = 0

            n_correct_top5 = 0
            n_samples_top5 = 0

            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                # top-1 accuracy
                _, prediction = torch.max(outputs, dim=1)
                n_samples += labels.shape[0]
                n_correct += (prediction == labels).sum().item()

                # top-5 accuracy
                _, predictions = torch.topk(input=outputs, k=5, dim=1)
                count = 0
                for pred in predictions:
                    n_samples_top5 += 1
                    n_correct_top5 += (pred in labels[count])
                    count += 1

        top1_accuracy = 100 * n_correct / n_samples
        top5_accuracy = 100 * n_correct_top5 / n_samples_top5

        return top1_accuracy, top5_accuracy

    def plot_and_save(self, model_name, save_folder):
        """
        _summary_

        Args:
            save_folder (_type_): _description_
        """
        # Plot losses
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs, self.losses, label='Loss', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training Loss')
        plt.legend()
        plt.grid(True)

        # Save the loss figure
        loss_figure_path = f"{save_folder}/training_loss.png"
        plt.savefig(loss_figure_path)
        plt.close()

        # Plot accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs, self.val1_acc, label='Top-1 Accuracy', color='green')
        plt.plot(self.epochs, self.val5_acc, label='Top-5 Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} Validation Accuracy')
        plt.legend()
        plt.grid(True)

        # Save the accuracy figure
        accuracy_figure_path = f"{save_folder}/validation_accuracy.png"
        plt.savefig(accuracy_figure_path)
        plt.close()


@dataclass
class AlexNetTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """
    def _get_output(self, images, labels):
        """
        _summary_

        Args:
            images (_type_): _description_
            labels (_type_): _description_
        """
        y_hat = self.model(images)
        loss = self.loss_fn(y_hat, labels)

        return loss


@dataclass
class BranchyNetTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """

    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    def _get_output(self, images, labels):
        """
        _summary_

        Args:
            images (_type_): _description_
            labels (_type_): _description_
        """
        y_hats = self.model(images)
        loss = 0
        for weight, y_branch in zip(self.weights, y_hats):
            loss += (weight * self.loss_fn(y_branch, labels))

        return loss
