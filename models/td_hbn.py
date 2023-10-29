"""
_summary_
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TD_HBN(nn.Module):
    """
    _summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, dropout: float = 0.5):
        super().__init__()

        self.adaptivepool = nn.AdaptiveAvgPool2d((6, 6))

        self.features1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.sem_complexity = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.LayerNorm(128),
            nn.Linear(128, 2),
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
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )

        # exit 1
        self.exit1a = nn.Sequential(
            nn.Linear(2048, 100)
        )

        self.exit1b = nn.Sequential(
            nn.Linear(2048, 20)
        )

        self.features2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        self.lin2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )

        # exit 2
        self.exit2a = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 100),
        )

        self.exit2b = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 20)
        )

        self.features3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
        )

        # exit 3
        self.exit3a = nn.Sequential(
            nn.Linear(2048, 100),
        )

        self.exit3b = nn.Sequential(
            nn.Linear(2048, 20),
        )

    def forward(self, x: torch.Tensor, threshold: list = None):
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
        z1_ = self.adaptivepool(z1_)
        z1_lin = self.lin1(z1_.view(z1_.size(0), -1))
        z1_fine, z1_coarse, z_fine_entropy, z_coarse_entropy = self.evaluate_output(z1_lin,
                                                  self.exit1a, self.exit1b)

        # get image semantic complexity
        sem = self.adaptivepool(a1)
        sem_granularity = self.sem_complexity(sem.view(sem.size(0), -1))

        if not self.training:
            _, granularities = torch.max(sem_granularity, 1)
            for granularity, z_fine_ent, z_coarse_ent in zip(granularities,
                                                            z_fine_entropy, z_coarse_entropy):
                if (granularity == 1) and (z_fine_ent < threshold[0]):
                    # -> fine classification
                    return z1_fine, 'fine exit 1'
                elif (granularity == 0) and (z_coarse_ent < threshold[0]):
                    # -> coarse classification
                    return z1_coarse, 'coarse exit 1'

        a2 = self.features2(a1)
        z2_ = self.branch2(a2)
        z2_ = self.adaptivepool(z2_)
        z2_lin = self.lin2(z2_.view(z2_.size(0), -1))
        z2_fine, z2_coarse, z_fine_entropy, z_coarse_entropy = self.evaluate_output(z2_lin,
                                                  self.exit2a, self.exit2b)

        if not self.training:
            for granularity, z_fine_ent, z_coarse_ent in zip(granularities,
                                                            z_fine_entropy, z_coarse_entropy):
                if (granularity == 1) and (z_fine_ent < threshold[1]):
                    # -> fine classification
                    return z2_fine, 'fine exit 2'
                elif (granularity == 0) and (z_coarse_ent < threshold[1]):
                    # -> coarse classification
                    return z2_coarse, 'coarse exit 2'

        a3 = self.features3(a2)
        z3_ = self.adaptivepool(a3)
        z3_lin = self.classifier(z3_.view(z3_.size(0), -1))
        z3_fine, z3_coarse, z_fine_entropy, z_coarse_entropy = self.evaluate_output(z3_lin,
                                                  self.exit3a, self.exit3b)

        if not self.training:
            for granularity, z_fine_ent, z_coarse_ent in zip(granularities,
                                                            z_fine_entropy, z_coarse_entropy):
                if granularity == 1:
                    # -> fine classification
                    return z3_fine, 'fine exit 3'
                else:
                    # -> coarse classification
                    return z3_coarse, 'coarse exit 3'

        # return training data
        return z1_fine, z1_coarse, z2_fine, z2_coarse, z3_fine, z3_coarse, sem_granularity

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
