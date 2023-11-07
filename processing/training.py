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
    pt_optimizer: torch.optim.Optimizer
    args: argparse.Namespace
    device: torch.device
    g1_identifiers: list
    epochs: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    branch_losses: List[float] = field(default_factory=list)
    sem_losses: List[float] = field(default_factory=list)
    val1_acc: List[float] = field(default_factory=list)
    val5_acc: List[float] = field(default_factory=list)
    specificities: List[float] = field(default_factory=list)
    # want to maximize early exits, therefore, they have higher weights in the loss
    weights: List[float] = field(default_factory=lambda: [1.0, 0.9, 0.8])

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
            loss_temp = []
            for index, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                loss, _, _ = self._get_output(images, labels)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_temp.append(loss.item())

                if (index + 1) % num_iterations == 0:
                    print(f"Epoch [{epoch + 1}/{self.args.num_epochs}]:")
                    top1_val, top5_val, specificity = self.validate()
                    print(f"\t\tTop 1 Acc = {top1_val}%\n\t\tTop 5 Acc = {top5_val}% \
                                \n\t\tLoss/Iteration: {sum(loss_temp) / len(loss_temp)}\n")

                    # add data to lists
                    self.epochs.append(epoch+1)
                    self.losses.append(sum(loss_temp) / len(loss_temp))
                    self.val1_acc.append(top1_val)
                    self.val5_acc.append(top5_val)
                    self.specificities.append(sum(specificity) / len(specificity))

                    if (epoch + 1) % 10 == 0:
                        # save checkpoint
                        torch.save(self.model.state_dict(), filepath)

                    # clear variables that are no longer needed
                    del images
                    del labels
                    del loss
                    del top1_val
                    del top5_val
                    del specificity

        self._post_training()

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

            specificity = []

            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # get new labels
                new_labels = [self.g1_identifiers[i.item()] for i in labels]
                new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

                outputs, exits = self._get_output(images, labels)

                if exits and 'coarse' in exits:
                    labels = new_labels
                    specificity.append(0)
                else:
                    specificity.append(1)

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

        print(f"\t\tSpecificity: {sum(specificity) / len(specificity)}")

        top1_accuracy = 100 * n_correct / n_samples
        top5_accuracy = 100 * n_correct_top5 / n_samples_top5

        return top1_accuracy, top5_accuracy, specificity

    @abstractmethod
    def _post_training(self):
        """
        _summary_
        """

    def plot_and_save(self, model_name, save_folder):
        """
        _summary_

        Args:
            save_folder (_type_): _description_
        """

        # plot loss and save
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(self.epochs, self.losses, label='Loss', color='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Specificity')
        ax2.plot(self.epochs, self.specificities, label='Specificity', color='red')

        plt.title(f'{model_name} Training Loss')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')

        fig.tight_layout()
        loss_figure_path = f"{save_folder}/training_loss.png"
        plt.savefig(loss_figure_path)
        plt.close()

        # plot accuracy and save
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs, self.val1_acc, label='Top-1 Accuracy', color='green')
        plt.plot(self.epochs, self.val5_acc, label='Top-5 Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} Validation Accuracy')
        plt.grid(True)

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

            return loss, None, None
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _post_training(self):
        pass


@dataclass
class BranchyNetTrainer(Trainer):
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
            loss = 0
            y_hats = self.model(images)
            for weight, y_branch in zip(self.weights, y_hats):
                loss += weight * self.loss_fn(y_branch, labels)

            return loss, None, None
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _post_training(self):
        pass


@dataclass
class SemHBNTrainer(Trainer):
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
            # get new labels
            new_labels = [self.g1_identifiers[i.item()] for i in labels]
            new_labels = torch.Tensor(new_labels).type(torch.LongTensor).to(self.device)

            y_hats = self.model(images)
            loss = 0
            for idx, (y_branch) in enumerate(y_hats):
                if y_branch.size(1) == 20:
                    loss += self.weights[idx//2] * self.loss_fn(y_branch, new_labels)
                elif y_branch.size(1) == 100:
                    loss += self.weights[idx//2] * self.loss_fn(y_branch, labels)

            return loss, loss, None
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _post_training(self):
        pass


@dataclass
class SuperNetTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """
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

            return loss, loss, None
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _post_training(self):
        pass


@dataclass
class TD_HBNTrainer(Trainer):
    """
    _summary_

    Args:
        Trainer (_type_): _description_
    """
    # semantic weighting to equalise the value of the losses
    sem_weighting: float = 20.0
    fine_weighting: float = 1.35

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

            # use branch loss during training and semantic loss during post-training
            loss = branch_loss
            return loss, branch_loss, semantic_loss
        elif not self.model.training:
            y_hat, exits = self.model(images)
            return y_hat, exits

    def _get_sem_loss(self, sem_granularity, y_hats, labels, new_labels):
        # GET THE BEST GRANULARITY BASED ON THE FINE AND COARSE LOSSES #
        coarse_right, fine_right = [], []
        for y_hat in y_hats:
            _, prediction = torch.max(y_hat, axis=1)
            if y_hat.size(1) == 20:
                coarse_right.append(prediction == new_labels)
            elif y_hat.size(1) == 100:
                fine_right.append(prediction == labels)

        # find the optimal granularity
        fine_right_value = torch.stack(fine_right).int().sum(dim=0) * self.fine_weighting
        coarse_right_value = torch.stack(coarse_right).int().sum(dim=0)

        opt_value = (fine_right_value > coarse_right_value).type(torch.long)

        # get classification error
        sem_loss = self.loss_fn(sem_granularity, opt_value)
        return sem_loss

    def _post_training(self):
        num_iterations = len(self.train_loader)

        print(f"\n# Post-Training iterations per epoch : {num_iterations}\n")
        print("-"*40 + "\n|" + " "*13 + "Post-Training" + " "*12 + "|\n" + "-"*40 + "\n")
        for epoch in range(self.args.num_epochs // 3):
            loss_temp = []
            for index, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                _, _, semantic_loss = self._get_output(images, labels)

                # backward pass
                self.pt_optimizer.zero_grad()
                semantic_loss.backward()
                self.pt_optimizer.step()
                loss_temp.append(semantic_loss.item())

                if (index + 1) % num_iterations == 0:
                    print(f"Epoch [{epoch + 1}/{self.args.num_epochs // 3}]:")
                    top1_val, top5_val, specificity = self.validate()
                    print(f"\t\tTop 1 Acc = {top1_val}%\n\t\tTop 5 Acc = {top5_val}% \
                                \n\t\tLoss/Iteration: {sum(loss_temp) / len(loss_temp)}\n")

    def investigating_post_trainer(self):
        """
        _summary_
        """
        # load the model
        self.model.load_state_dict(torch.load('../results/models/TD-HBN-trained.pth'))

        num_iterations = len(self.train_loader)

        print(f"\n# Post-Training iterations per epoch : {num_iterations}\n")
        print("-"*40 + "\n|" + " "*13 + "Post-Training" + " "*12 + "|\n" + "-"*40 + "\n")
        for epoch in range(self.args.num_epochs // 3):
            loss_temp = []
            for index, (images, labels) in enumerate(self.train_loader):
                self.model.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # forward pass
                _, _, semantic_loss = self._get_output(images, labels)

                # backward pass
                self.pt_optimizer.zero_grad()
                semantic_loss.backward()
                self.pt_optimizer.step()
                loss_temp.append(semantic_loss.item())

                if (index + 1) % num_iterations == 0:
                    print(f"Epoch [{epoch + 1}/{self.args.num_epochs // 3}]:")
                    top1_val, top5_val, specificity = self.validate()
                    print(f"\t\tTop 1 Acc = {top1_val}%\n\t\tTop 5 Acc = {top5_val}% \
                                \n\t\tLoss/Iteration: {sum(loss_temp) / len(loss_temp)}\n")
                    # save checkpoint
                    torch.save(self.model.state_dict(), '../results/models/TD-HBN-post-trained.pth')
