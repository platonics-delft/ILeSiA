
from risk_estimation.models.risk_estimator import RiskEstimatorBase
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pathlib

class ResNetRiskEstimator(RiskEstimatorBase):
    APPROACH = "ResNet-50"

    def __init__(self,
                 name: str,
                 batch_size: int = 40, 
                 thr: float = 0.5,
                 learning_rate: float = 0.01,
                 arch: str = "resnet-50",
                 train_patience: int = 3,
                 train_epoch: int = 10,
                 ):
        super(ResNetRiskEstimator, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.thr = thr
        self.learning_rate = learning_rate
        self.arch = arch
        self.patience = train_patience
        self.train_epoch = train_epoch

    def load_model(self):
        print(f"Loading Risk Estimation model: {self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")
        checkpoint = torch.load(f"{self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")

        self.create_model(checkpoint['X'], checkpoint['Y'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.move_model_to_cuda()

    def save_model(self):
        """Overloaded function, saves also ra_model
        """        
        pathlib.Path(f"{self.model_path}").mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(), 
            # "X": self.dataloader.dataset.X,
            # "Y": self.dataloader.dataset.Y,
            }, f"{self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(), 
            # "X": self.dataloader.dataset.X,
            # "Y": self.dataloader.dataset.Y,
            }, f"{self.model_path}/{self.name}_{self.encode_params_as_str()}_model.pt")

    def create_model(self):        
        if self.arch == 'resnet-50':
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        else: raise Exception()


    def move_model_to_cuda(self):
        self.model=self.model.cuda()

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


    def training_loop(self, dataloader, early_stop=False):
        self.dataloader =dataloader

        dataloader, validation_dataloader = self.split_dataloader(dataloader)
        
        self.create_model()
        self.move_model_to_cuda()

        # X_train, Y_train = RiskEstimationDataset.dataloader_to_array(dataloader)
        # Y_train = Y_train.cpu().numpy().squeeze()
        # X_validation, Y_validation = RiskEstimationDataset.dataloader_to_array(validation_dataloader)
        # Y_validation = Y_validation.cpu().numpy().squeeze()

        # if self.dataloader_test_for_plot is not None:
        #     X_test, Y_test = RiskEstimationDataset.dataloader_to_array(self.dataloader_test_for_plot)
        #     Y_test = Y_test.cpu().numpy().squeeze()
        #     X_nodrop, Y_nodrop = RiskEstimationDataset.dataloader_to_array(self.dataloader_nodrop_for_plot)
        #     Y_nodrop = Y_nodrop.cpu().numpy().squeeze()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.model.train()
        for epoch in range(self.train_epoch):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                output = self.model(inputs.cuda())
                labels = torch.tensor(labels, dtype=torch.float32)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def sample(self, X):
        if X.ndim == 1:
            X = X[None, X]
        # assert X.ndim == 2

        self.model.eval()
        with torch.no_grad():
            
            outputs = np.array([self.model(X[i:i+1]).cpu().numpy()[0] for i in range(len(X))])

            pred = np.array([np.argmax((o[0], o[1])) for o in outputs]) 
            
            # print(f"observed variance: {std}")
            # pred = self.risk_to_decision(risk)
        
        self.model.train()
        return pred, pred, 0.0
