


import pathlib
import gpytorch
from risk_estimation.models.gaussian_process.gp_classifier import GPModel
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from scipy.spatial.distance import cosine
import numpy.typing as npt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
import error_recovery
from video_embedding.utils import get_session

class RecoveryStateFinder():
    def __init__(self):
        super(RecoveryStateFinder, self).__init__()
        # self.dataloader_test_for_plot = None
        # self.dataloader_nodrop_for_plot = None
        self.trained_epoch = 0  # for how many epoch was model trained

    @property
    def model_path(self):
        return f"{error_recovery.path}/saved_models/{get_session()}"

    def sample(self, h: npt.NDArray[np.float64]) -> float:
        """ Given latent configuration h, returns trajectory phase
        """
        return 0.0, 0.0

        # alpha = h[0][-1]
        # if alpha < 0.7:
        # else: 
        #     return 0.99, 0.0

class GPRecoveryStateFinder(RecoveryStateFinder):
    APPROACH = "GP"
    def __init__(self,
                name: str, 
                xdim: int = None,
                batch_size: int = 40,
                ard: bool = True,
                learning_rate: float = 0.01,
                arch: str = "",
                train_patience: int = 3000,
                ):
        self.name = name
        self.batch_size = batch_size
        self.xdim = xdim
        self.ard = ard
        self.learning_rate = learning_rate
        self.arch = arch
        self.patience = train_patience



    def sample(self, X):
        if X.ndim == 1:
            X = X[None, X]
        assert X.ndim == 2

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            # alpha = self.model.forward(X).cpu().detach().numpy().ravel()
            alpha = observed_pred.mean.cpu().numpy()
            std = observed_pred.stddev.cpu().numpy()

        self.model.train()
        self.likelihood.train()

        return alpha, std

    def load_model(self):
        print(f"Loading Error Recovery model: {self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")
        checkpoint = torch.load(f"{self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")

        self.create_model(checkpoint['X'], checkpoint['Y'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.move_model_to_cuda()

    def save_model(self):
        pathlib.Path(f"{self.model_path}").mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(), 
            "X": self.model.train_x,
            "Y": self.model.train_y,
            }, f"{self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")
        
    def create_model(self, X, Y):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        print("Has analytical likelihood:")
        print(self.likelihood.has_analytic_marginal)

        if self.arch == '':
            self.model = GPModel(X, Y, self.likelihood, ard=self.ard)
        elif self.arch == 'L+GP':
            self.model = GPModelPrepro(X, Y, self.likelihood, ard=self.ard)
        elif self.arch == 'L+GP+1SKIP':
            self.model = GPModelPreproFeatureSkip(X, Y, self.likelihood, ard=self.ard, n_features_to_skip=1)
        elif self.arch == 'L+GP+2SKIP':
            self.model = GPModelPreproFeatureSkip(X, Y, self.likelihood, ard=self.ard, n_features_to_skip=2)
        else: raise Exception()

    def move_model_to_cuda(self):
        self.model=self.model.cuda()
        self.likelihood=self.likelihood.cuda()

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


    def training_loop(self, dataloader):

        dataloader, validation_dataloader = self.split_dataloader(dataloader)

        X, Y = RiskEstimationDataset.dataloader_to_array(dataloader)
        Y = Y.squeeze()

        self.create_model(X, Y)
        self.move_model_to_cuda()

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=self.learning_rate)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        X_train, Y_train = RiskEstimationDataset.dataloader_to_array(dataloader)
        Y_train = Y_train.cpu().numpy().squeeze()
        X_validation, Y_validation = RiskEstimationDataset.dataloader_to_array(validation_dataloader)
        Y_validation = Y_validation.cpu().numpy().squeeze()

        if self.dataloader_test_for_plot is not None:
            X_test, Y_test = RiskEstimationDataset.dataloader_to_array(self.dataloader_test_for_plot)
            Y_test = Y_test.cpu().numpy().squeeze()
            X_nodrop, Y_nodrop = RiskEstimationDataset.dataloader_to_array(self.dataloader_nodrop_for_plot)
            Y_nodrop = Y_nodrop.cpu().numpy().squeeze()


        
        # early_stopping = GPEarlyStopping(self.patience)
        epochs_iter = tqdm(range(self.train_epoch))
        for i in epochs_iter:
   
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, Y)
            loss.backward()
            optimizer.step()
            
            
            # if i%1 == 0:
            #     Y_pred, _ = self.sample(X_train)
            #     acc_train =  100 * (Y_train == Y_pred).mean()
            #     Y_pred, _ = self.sample(X_validation)
            #     acc_validation =  100 * (Y_validation == Y_pred).mean()

            #     acc_test = None
            #     acc_nodrop = None
            #     if self.dataloader_test_for_plot is not None:
            #         Y_pred, _ = self.sample(X_test)
            #         acc_test =  100 * (Y_test == Y_pred).mean()
            #         Y_pred, _ = self.sample(X_nodrop)
            #         acc_nodrop =  100 * (Y_nodrop == Y_pred).mean()


            #    # if early_stopping(i, acc_train, acc_validation, acc_test, acc_nodrop):
            #    #     break
    
        self.trained_epoch = i
        print("Training finished")
        # print("Lengthscale: ")
        # print(self.model.covar_module.base_kernel.lengthscale)
        # print("Outputscale: ")
        # print(torch.sqrt(self.model.covar_module.outputscale))
    
