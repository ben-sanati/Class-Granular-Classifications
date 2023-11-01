"""
_summary_
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SuperHBN(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_fine_classes: int = 100, num_coarse_classes: int = 20,
                 dropout: float = 0.5):
        super().__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        # branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.lin1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )

        # exit 1
        self.exit1a = nn.Sequential(
            nn.Linear(2048, num_fine_classes)
        )

        self.exit1b = nn.Sequential(
            nn.Linear(2048, num_coarse_classes)
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.lin2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 1 * 1, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # exit 2
        self.exit2a = nn.Sequential(
            nn.Linear(2048, num_fine_classes),
        )

        self.exit2b = nn.Sequential(
            nn.Linear(2048, num_coarse_classes),
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        # exit 3
        self.exit3a = nn.Sequential(
            nn.Linear(4096, num_fine_classes),
        )

        self.exit3b = nn.Sequential(
            nn.Linear(4096, num_coarse_classes),
        )

    def forward(self, x: torch.Tensor, threshold: list = None, fine_tolerance: float = 0.5):
        """
        _summary_

        Args:
            x (torch.Tensor): _description_
            threshold (list, optional): _description_. Defaults to None.
            fine_tolerance (float, optional): _description_. Defaults to 0.5.

        Returns:
            _type_: _description_
        """
        # set default threshold values if not provided
        if threshold is None:
            threshold = [0.5, 0.5]

        # run branch 1
        a1 = self.features1(x)
        z1_ = self.branch1(a1)
        z1_lin = self.lin1(z1_.view(z1_.size(0), -1))
        z1_fine, z1_coarse, z_fine_entropy, z_coarse_entropy = \
                        self.evaluate_output(z1_lin, self.exit1a, self.exit1b)

        if not self.training:
            if (z_fine_entropy.mean() - fine_tolerance < z_coarse_entropy.mean() and
                    z_fine_entropy.mean() < threshold[0]):
                return z1_fine, 'fine exit 1'
            elif (z_coarse_entropy.mean() < z_fine_entropy.mean() - fine_tolerance and
                    z_coarse_entropy.mean() < threshold[0]):
                return z1_coarse, 'coarse exit 1'

        a2 = self.features2(a1)
        z2_ = self.branch2(a2)
        z2_lin = self.lin2(z2_.view(z2_.size(0), -1))
        z2_fine, z2_coarse, z_fine_entropy, z_coarse_entropy = \
                        self.evaluate_output(z2_lin, self.exit2a, self.exit2b)

        if not self.training:
            if (z_fine_entropy.mean() - fine_tolerance < z_coarse_entropy.mean() and
                    z_fine_entropy.mean() < threshold[1]):
                return z2_fine, 'fine exit 2'
            elif (z_coarse_entropy.mean() < z_fine_entropy.mean() - fine_tolerance and
                    z_coarse_entropy.mean() < threshold[1]):
                return z2_coarse, 'coarse exit 2'

        a3 = self.features3(a2)
        z3_lin = self.classifier(z3_.view(z3_.size(0), -1))
        z3_fine, z3_coarse, z_fine_entropy, z_coarse_entropy = \
                            self.evaluate_output(z3_lin, self.exit3a, self.exit3b)

        if not self.training:
            if z_fine_entropy.mean() - fine_tolerance < z_coarse_entropy.mean():
                return z3_fine, 'fine exit 3'
            else:
                return z3_coarse, 'coarse exit 3'

        # return training data
        return z1_fine, z1_coarse, z2_fine, z2_coarse, z3_fine, z3_coarse

    def evaluate_output(self, x, exit_a, exit_b):
        """
        _summary_

        Args:
            x (_type_): _description_
            exit_a (_type_): _description_
            exit_b (_type_): _description_

        Returns:
            _type_: _description_
        """
        z_fine = exit_a(x)
        z_coarse = exit_b(x)

        z_fine_probs = F.softmax(z_fine, dim=1)
        z_coarse_probs = F.softmax(z_coarse, dim=1)

        z_fine_entropy = Categorical(z_fine_probs).entropy()
        z_coarse_entropy = Categorical(z_coarse_probs).entropy()

        return z_fine, z_coarse, z_fine_entropy, z_coarse_entropy
