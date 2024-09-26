from typing import Iterable, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from risk_estimation.models.risk_estimation.frame_dropping import (
    NoFrameDroppingPolicy,
)

from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import load



class RiskClassifierA1(nn.Module):
    def __init__(self, xdim):
        super(RiskClassifierA1, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)

class RiskClassifierA2(nn.Module):
    def __init__(self, xdim):
        super(RiskClassifierA2, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)

class RiskClassifierA3(nn.Module):
    def __init__(self, xdim):
        super(RiskClassifierA3, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)


class RiskClassifierA4(nn.Module):
    def __init__(self, xdim):
        super(RiskClassifierA4, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)

class RiskClassifierA5(nn.Module):
    def __init__(self, xdim):
        super(RiskClassifierA5, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)

class LRRiskClassifier(nn.Module):
    def __init__(self, xdim):
        super(LRRiskClassifier, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)


