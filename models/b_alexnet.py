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

        self.adaptivepool = nn.AdaptiveAvgPool2d((6, 6))

        # b1
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        self.exit1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 100),
        )

        # b2
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        self.exit2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 100),
            nn.ReLU(inplace=True)
        )

        # b3
        self.features3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor, threshold: list = None):
        """
        _summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # Set default threshold values if not provided
        if threshold is None:
            threshold = [0.5, 0.5, 0.5]

        # Pass input through the first set of layers
        a1 = self.features1(x)
        z1_ = self.branch1(a1)
        z1_ = self.adaptivepool(z1_)
        z1 = self.exit1(z1_.view(z1_.size(0), -1))

        # Calculate entropy and check for early exit
        z1_probs = F.softmax(z1, dim=1)
        z1_entropy = Categorical(z1_probs).entropy()

        if not self.training and z1_entropy.mean() < threshold[0]:
            return z1

        # Pass input through the second set of layers
        a2 = self.features2(a1)
        z2_ = self.branch2(a2)
        z2_ = self.adaptivepool(z2_)
        z2 = self.exit2(z2_.view(z2_.size(0), -1))

        # Calculate entropy and check for early exit
        z2_probs = F.softmax(z2, dim=1)
        z2_entropy = Categorical(z2_probs).entropy()

        if not self.training and z2_entropy.mean() < threshold[1]:
            return z2

        # Pass input through the third set of layers and return the final output
        a3 = self.features3(a2)
        a3 = self.adaptivepool(a3)
        z3 = self.classifier(a3.view(a3.size(0), -1))

        if self.training:
            return z1, z2, z3
        else:
            return z3
