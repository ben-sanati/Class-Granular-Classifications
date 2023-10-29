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
import torch.nn.functional as F
from torch.distributions import Categorical


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
    g1_identifiers: list
    epochs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    branch_losses: List[float] = field(default_factory=list)
    sem_losses: List[float] = field(default_factory=list)
    val1_acc: List[float] = field(default_factory=list)
    val5_acc: List[float] = field(default_factory=list)

    def init_param(self):
        """
        Kaiming Normal parameter initialization
        -------------------------------------------------------------------------------
        Ref:
        He, K., Zhang, X., Ren, S. and Sun, J., 2015. Delving deep into rectifiers:
        Surpassing human-level performance on imagenet classification. In Proceedings
        of the IEEE international conference on computer vision (pp. 1026-1034)
        -------------------------------------------------------------------------------
        """
        if isinstance(self.model, nn.Conv2d):
            nn.init.kaiming_normal_(self.model.weight.data, nonlinearity='relu')
            nn.init.constant_(self.model.bias.data, 0)
        elif isinstance(self.model, nn.Linear):
            nn.init.xavier_normal_(self.model.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.model.bias.data, 0)

    def train(self, filepath: str):
        """
        _summary_
        """
        num_iterations = len(self.train_loader)

        print(f"\n# Training iterations per epoch : {num_iterations}\n")
        print("-"*30 + "\n|" + " "*10 + "Training" + " "*10 + "|\n" + "-"*30 + "\n")
        for epoch in range(self.args.num_epochs):
            loss_temp, branch_temp, semantic_temp = [], [], []
            for index, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                loss, branch_loss, semantic_loss = self._get_output(images, labels)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_temp.append(loss.item())
                branch_temp.append(branch_loss.item())
                semantic_temp.append(semantic_loss.item())

                if (index + 1) % num_iterations == 0:
                    top1_val, top5_val = self.validate()
                    print(f"Epoch [{epoch + 1}/{self.args.num_epochs}]: \
                            \n\t\tTop 1 Acc = {top1_val}%\n\t\tTop 5 Acc = {top5_val}% \
                                \n\t\tLoss/Iteration: {sum(loss_temp) / len(loss_temp)} \
                                    \n\t\tBranch Loss: {sum(branch_temp) / len(branch_temp)} \
                                        \n\t\tSemantic Loss: {sum(semantic_temp) / len(semantic_temp)}\n")

                    # add data to lists
                    self.epochs.append(epoch+1)
                    self.losses.append(sum(loss_temp) / len(loss_temp))
                    self.branch_losses.append(sum(branch_temp) / len(branch_temp))
                    self.sem_losses.append(sum(semantic_temp) / len(semantic_temp))
                    self.val1_acc.append(top1_val)
                    self.val5_acc.append(top5_val)

                    # clear variables that are no longer needed
                    del images
                    del labels
                    del loss
                    del top1_val
                    del top5_val

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

                # get new labels
                new_labels = [self.g1_identifiers[i.item()] for i in labels]
                new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

                outputs, exits = self._get_output(images, labels)

                if exits and 'coarse' in exits:
                    labels = new_labels

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

        if self.branch_losses:
            plt.plot(self.epochs, self.branch_losses, label='Branch Loss', color='green')
        if self.sem_losses:
            plt.plot(self.epochs, self.sem_losses, label='Semantic Loss', color='red')

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
        if self.model.training:
            y_hat, _ = self.model(images)
            loss = self.loss_fn(y_hat, labels)

            return loss
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits


@dataclass
class BranchyNetTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """
    # want to maximize early exits, therefore, they have higher weights in the loss
    weights: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.4])

    def _get_output(self, images, labels):
        """
        _summary_

        Args:
            images (_type_): _description_
            labels (_type_): _description_
        """
        if self.model.training:
            loss = 0
            y_hats = self.model(images)
            for weight, y_branch in zip(self.weights, y_hats):
                loss += weight * self.loss_fn(y_branch, labels)

            return loss
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits


@dataclass
class SuperNetTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """

    # branch weights are the same as in BranchyNet
    weights: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.4])
    # semantic weights vary on what we prefer -> fine classes
    fine_class_weighting: float = 1.5
    coarse_class_weighting: float = 0.75

    def _get_output(self, images, labels):
        """
        _summary_

        Args:
            images (_type_): _description_
            labels (_type_): _description_
        """
        if self.model.training:
            # get new labels
            new_labels = [self.g1_identifiers[i.item()] for i in labels]
            new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

            y_hats = self.model(images)
            loss = 0
            for idx, (y_branch) in enumerate(y_hats):
                if y_branch.size(1) == 20:
                    loss += (self.weights[idx//2] * self.coarse_class_weighting *
                             self.loss_fn(y_branch, new_labels))
                elif y_branch.size(1) == 100:
                    loss += (self.weights[idx//2] * self.fine_class_weighting *
                             self.loss_fn(y_branch, labels))

            return loss
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits


@dataclass
class TD_HBNTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """

    # branch weights are the same as in BranchyNet
    weights: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.4])
    # semantic weighting to equalise the value of the losses
    sem_weighting: float = 20.0

    def _get_output(self, images, labels):
        """
        _summary_

        Args:
            images (_type_): _description_
            labels (_type_): _description_
        """
        if self.model.training:
            # get new labels
            new_labels = [self.g1_identifiers[i.item()] for i in labels]
            new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

            output = self.model(images)
            y_hats = output[:-1]
            sem_granularity = output[-1]

            # get branch loss
            branch_loss = 0
            losses = {'coarse': [], 'fine': []}
            for idx, (y_branch) in enumerate(y_hats):
                if y_branch.size(1) == 20:
                    coarse_loss = (self.weights[idx//2] *
                                     self.loss_fn(y_branch, new_labels))
                    branch_loss += coarse_loss

                    losses['coarse'].append(coarse_loss)
                elif y_branch.size(1) == 100:
                    fine_loss = (self.weights[idx//2] *
                                     self.loss_fn(y_branch, labels))
                    branch_loss += fine_loss

                    losses['fine'].append(fine_loss)

            # get semantic loss
            semantic_loss = self._get_sem_loss(sem_granularity,
                                               y_hats, labels, new_labels) * self.sem_weighting

            # sum total losses
            loss = branch_loss + semantic_loss
            return loss, branch_loss, semantic_loss
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _get_sem_loss(self, sem_granularity, y_hats, labels, new_labels):
        # get the best granularity based the fine and coarse losses
        coarse_entropies, fine_entropies = [], []
        for y_hat in y_hats:
            if y_hat.size(1) == 20:
                pred_probs = F.softmax(y_hat, dim=1)
                pred_entropy = Categorical(pred_probs).entropy()
                coarse_entropies.append(pred_entropy)
            if y_hat.size(1) == 100:
                pred_probs = F.softmax(y_hat, dim=1)
                pred_entropy = Categorical(pred_probs).entropy()
                fine_entropies.append(pred_entropy)

        coarse_entropies = torch.stack(coarse_entropies)
        coarse_entropies_ave = torch.mean(coarse_entropies, dim=0)
        fine_entropies = torch.stack(fine_entropies)
        fine_entropies_ave = torch.mean(fine_entropies, dim=0)

        # get the optimal granularity based on the current image
        # leads to varying fine-tolerance values based on the current difference in entropies
        fine_tolerance = (fine_entropies_ave - coarse_entropies_ave).mean()

        # we can now define the optimal sem_granularity matrix
        opt_sem_granularity = fine_entropies_ave - coarse_entropies_ave - fine_tolerance

        # if opt_sem_granularity is positive -> fine entropy is large -> uncertain -> use coarse
        # if opt_sem_granularity is negative -> fine entropy is small -> certain -> use fine
        # if opt_sem_granularity is positive -> set matrix element to 0
        # if opt_sem_granularity is negative -> set matrix element to 1
        positive_mask = opt_sem_granularity > 0
        negative_mask = opt_sem_granularity < 0
        opt_sem_granularity[positive_mask] = 0
        opt_sem_granularity[negative_mask] = 1
        opt_sem_granularity = opt_sem_granularity.type(torch.long)

        # cross-entropy loss between the opt_granularity and the sem_granularity
        sem_loss = self.loss_fn(sem_granularity, opt_sem_granularity)
        return sem_loss
