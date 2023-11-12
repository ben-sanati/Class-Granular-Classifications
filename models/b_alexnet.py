"""
_summary_
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class BranchyAlexNet(nn.Module):
    """
    _summary_
    """
    def __init__(self, num_classes: int = 100, dropout: float = 0.5) -> None:
        super().__init__()

        # features => AlexNet backbone
        # branches => branches off backbone to early exits
        # exits => early linear classifiers

        # b1
        self.features1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.exit1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 2 * 2, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
        )

        # b2
        self.features2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        self.exit2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 1 * 1, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

        # b3
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
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor, threshold: list = None):
        """
        _summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # set default threshold values if not provided
        if threshold is None:
            threshold = [0.5, 0.5]

        # pass input through the first set of layers
        a1 = self.features1(x)
        z1_ = self.branch1(a1)
        z1 = self.exit1(z1_.view(z1_.size(0), -1))

        # calculate entropy and check for early exit
        z1_probs = F.softmax(z1, dim=1)
        z1_entropy = Categorical(z1_probs).entropy()

        if not self.training and z1_entropy.mean() < threshold[0]:
            return z1, 'exit 1'

        # pass input through the second set of layers
        a2 = self.features2(a1)
        z2_ = self.branch2(a2)
        z2 = self.exit2(z2_.view(z2_.size(0), -1))

        # calculate entropy and check for early exit
        z2_probs = F.softmax(z2, dim=1)
        z2_entropy = Categorical(z2_probs).entropy()

        if not self.training and z2_entropy.mean() < threshold[1]:
            return z2, 'exit 2'

        # pass input through the third set of layers and return the final output
        a3 = self.features3(a2)
        z3 = self.classifier(a3.view(a3.size(0), -1))

        if self.training:
            return z1, z2, z3
        else:
            return z3, 'exit 3'
