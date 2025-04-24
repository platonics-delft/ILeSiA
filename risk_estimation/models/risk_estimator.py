
from typing import Any
import risk_estimation
from video_embedding.utils import get_session
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np  
from collections import deque 


class RiskEstimatorBase():
    def __init__(self):
        super(RiskEstimatorBase, self).__init__()
        self.validation_dataloaders = {}
        self.trained_epoch = 0
        self.feature_extractor = None
        self.train_epoch = 2000

    def set_dataloaders_for_validation(self, list_of_dataloaders: list, names: list):
        self.validation_dataloaders = {}
        for dataloader, name in zip(list_of_dataloaders,names):
            self.validation_dataloaders[name] = dataloader

    @property
    def model_path(self):
        return f"{risk_estimation.path}/saved_models/{get_session()}"

    WINDOW_SIZE = 3
    prev = deque([False] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
    def risk_to_decision(self, r: float) -> int:
        """ Activation Logic
        Args:
            r (float): Risk Score
            thr (float, optional): Threshold of decision

        Returns:
            int: Risk Flag (0 safe or 1 risk)
        """
        # return np.array(prob > self.thr, dtype=int) # old
        d_ = list(self.prev) + list(np.array(r) > self.thr)
        ret = []
        for i in range(self.WINDOW_SIZE+1, len(d_) + 1):
            window = d_[i - self.WINDOW_SIZE:i]
            ret.append(all(window))

        self.prev = deque(d_[-self.WINDOW_SIZE:], maxlen=self.WINDOW_SIZE)
    
        return np.array(ret)

    def sample(self, 
               X: torch.Tensor # 1D or 2D tensor
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise Exception()
        # if X.ndim == 1:
        #     X = X[None, X]
        # assert X.ndim == 2

        # risk = self.model.forward(X).cpu().detach().numpy().ravel()
        # return self.risk_to_decision(risk), risk, np.zeros(len(risk))
    
    def set_feature_extractor(self, feature_extractor: Any):
        self.feature_extractor = feature_extractor
    
    def encode_params_as_str(self) -> str:
        try:
            arch = self.arch
        except AttributeError:
            arch = ""
        try:
            lr = f"lr{self.learning_rate}"
        except AttributeError:
            lr = ""
        try:
            patience = f"ptnce{self.patience}"
        except AttributeError:
            patience = ""
        try:
            xdim = f"xdim{self.xdim}"
        except AttributeError:
            xdim = ""
        try:
            out_assessment = self.out_assessment
        except AttributeError:
            out_assessment = ""
        try:
            trained_epoch = f"ep{round(self.trained_epoch, -2)}"
        except AttributeError:
            trained_epoch = ""

        return f"{self.APPROACH}_{arch}_{out_assessment}_{xdim}_{patience}_{lr}_{trained_epoch}"

    def split_dataloader(self, dataloader):
        dataset = dataloader.dataset

        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=0.1, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Create DataLoader
        dataloader1 = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        dataloader2 = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        return dataloader1, dataloader2

