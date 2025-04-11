
from risk_estimation.models.risk_estimator import RiskEstimatorBase
from risk_estimation.models.mlp_risk_estimator import *
from risk_estimation.models.gp_risk_estimator import GPEarlyStoppingAndPlot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pathlib

class MLPRiskEstimator2(RiskEstimatorBase):
    APPROACH = "MLP2"
    def __init__(self, 
                 name: str,
                 xdim: int = 8,
                 batch_size: int = 40,
                 thr: float = 0.5,
                 learning_rate: float = 0.01,
                 train_patience: int = 3000,
                 train_epoch: int = 3000, 
                 ):
        super(MLPRiskEstimator2, self).__init__()
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BinaryClassifier(xdim)
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.model.to(self.device)

        self.batch_size = batch_size
        self.xdim = xdim
        self.thr = thr

        self.patience = train_patience
        self.train_epoch = train_epoch

    def training_loop(self, dataloader, l1_lambda=1e-5, early_stop=False):

        dataloader, validation_dataloader = self.split_dataloader(dataloader)
        early_stopping = GPEarlyStoppingAndPlot(self.patience, dataloader, validation_dataloader, self.validation_dataloaders)
        
        self.epochs_iter = tqdm(range(self.train_epoch))

        for epoch in self.epochs_iter:
            total_loss = 0

                

            for x, labels in dataloader:
                self.model.train() # Set to training mode (enables dropout)
                outputs = self.model(x)
                # Calculate BCE loss
                bce_loss = self.criterion(outputs, labels.squeeze().unsqueeze(1).float())
            
                # Add L1 regularization
                l1_reg = 0
                for param in self.model.parameters():
                    l1_reg += torch.sum(torch.abs(param))
                    # Total loss = BCE loss + L1 regularization
                    # (L2 regularization is already applied via weight_decay in optimizer)
                    total_loss_batch = bce_loss + l1_lambda * l1_reg
                # Backward and optimize
                self.optimizer.zero_grad()
                
                total_loss_batch.backward()
                self.optimizer.step()
                total_loss += bce_loss.item()
                # Track only the BCE loss for reporting


            if early_stop:
                    if epoch%5 == 0:
                        if early_stopping(epoch, self):
                            break

    def load_model(self):
        print(f"Risk estimation model: {self.model_path}/{self.name}_{self.__class__.__name__}_{self.xdim}_model.pt")
        self.model.load_state_dict(torch.load(f"{self.model_path}/{self.name}_{self.__class__.__name__}_{self.xdim}_model.pt"))
        self.model.eval()

    def save_model(self):
        """Overloaded function, saves also ra_model
        """
        pathlib.Path(f"{self.model_path}/").mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_path}/{self.name}_{self.__class__.__name__}_{self.xdim}_model.pt")

    def sample(self, x):
        self.model.eval() # Set to evaluation mode (disables dropout)
        with torch.no_grad():
            outputs = self.model(x).squeeze()
            predicted = (outputs > 0.5).float()

        return predicted.cpu().numpy(), outputs.cpu().numpy(), np.zeros(len(outputs))


class MLPRiskEstimator(RiskEstimatorBase):
    APPROACH = "MLP"

    def __init__(self, 
                 name: str,
                 xdim: int = 8,
                 batch_size: int = 40,
                 thr: float = 0.5,
                 arch: str = 'A4',
                 learning_rate: float = 0.01,
                 train_patience: int = 3000,
                 train_epoch: int = 3000, 
                 ):
        super(MLPRiskEstimator, self).__init__()
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.get_classifier_arcitecture(xdim,arch)
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()  # Loss function for risk estimator
        self.model.to(self.device)

        self.batch_size = batch_size
        self.xdim = xdim
        self.thr = thr

        self.patience = train_patience
        self.train_epoch = train_epoch

    def sample(self, 
               X: torch.Tensor # 1D or 2D tensor
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.ndim == 1:
            X = X[None, X]
        assert X.ndim == 2

        risk = self.model.forward(X).cpu().detach().numpy().ravel()
        return self.risk_to_decision(risk), risk, np.zeros(len(risk))

    def get_classifier_arcitecture(self, xdim, arch):
        if arch == 'LR':
            return LRRiskClassifier(xdim)
        elif arch == 'A4':
            return RiskClassifierA4(xdim)
        else: raise Exception()

    def training_loop(self, dataloader, early_stop=False):
        self.epochs_iter = tqdm(range(self.train_epoch))

        dataloader, validation_dataloader = self.split_dataloader(dataloader)

        early_stopping = GPEarlyStoppingAndPlot(self.patience, dataloader, validation_dataloader, self.validation_dataloaders)
        try:
            for i in self.epochs_iter:
                for inputs, labels in dataloader:
                    labels = torch.tensor(labels, dtype=torch.float32).cuda()
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    self.loss = self.criterion(outputs.squeeze(), labels.squeeze())
                    self.loss.backward()
                    self.optimizer.step()

                if early_stop:
                    if i%5 == 0:
                        if early_stopping(i, self):
                            break

        except KeyboardInterrupt:
            print("Stopping on interrupt")
        finally:
            print("Continuing with the rest of the program")

        early_stopping.plot_save(self.name, self)

        self.trained_epoch = i

    def load_model(self):
        print(f"Risk estimation model: {self.model_path}/{self.name}_{self.__class__.__name__}_{self.xdim}_model.pt")
        self.model.load_state_dict(torch.load(f"{self.model_path}/{self.name}_{self.__class__.__name__}_{self.xdim}_model.pt"))
        self.model.eval()

    def save_model(self):
        """Overloaded function, saves also ra_model
        """
        pathlib.Path(f"{self.model_path}/").mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.model_path}/{self.name}_{self.__class__.__name__}_{self.xdim}_model.pt")

import torch
import torch.nn as nn
from torch.nn import functional as F

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

class LRRiskClassifier(nn.Module):
    def __init__(self, xdim):
        super(LRRiskClassifier, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(xdim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.estimator(x)

# Define a binary classifier with dropout and regularization
class BinaryClassifier(nn.Module):
    def __init__(self, xdim, hidden_dim=32, dropout_rate=0.3):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(xdim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout3 = nn.Dropout(dropout_rate)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x