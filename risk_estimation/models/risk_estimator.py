
import pathlib
from typing import Any, Dict, Iterable

from risk_estimation.models.risk_estimation.saliency_map_generator import get_saliency_map_for_image
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
import pandas as pd
from sklearn.metrics import accuracy_score
import risk_estimation
from risk_estimation.models.gaussian_process.gp_classifier import GPModel, GPModelPrepro, GPModelPreproFeatureSkip
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_classifier import LRRiskClassifier, RiskClassifierA1, RiskClassifierA2, RiskClassifierA3, RiskClassifierA4, RiskClassifierA5
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import StampedLatentObservationsRiskLabels, LatentObservationsRiskLabels, LatentObservationsSafeLabels, FeatureExtractor, StampedDistLatentObservationsRiskLabels
from video_embedding.models.video_embedder import VideoEmbedder
from risk_estimation.plot_utils import plot_threshold_labelled
from video_embedding.utils import get_session, list_files_in_folder, load, load_video, save_models_index_list, save_video, save_video_index_list

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import numpy as np  
import cv2
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine

from tqdm import tqdm
import torch, torchvision
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

SAVE_GPU_SPACE = True

class MarkovianRiskEstimator():
    def __init__(self):
        super(MarkovianRiskEstimator, self).__init__()
        self.dataloader_test_for_plot = None
        self.dataloader_nodrop_for_plot = None
        self.trained_epoch = 0
        self.feature_extractor = None
        self.train_epoch = 2000

    @property
    def model_path(self):
        return f"{risk_estimation.path}/saved_models/{get_session()}"

    def risk_to_decision(self, prob: float) -> int:
        """
        Args:
            prob (float): Probability of riskiness
            thr (float, optional): Threshold of decision

        Returns:
            int: Decision (1 safe or 0 risk)
        """
        return np.array(prob > self.thr, dtype=int)

    def sample(self, X):
        if X.ndim == 1:
            X = X[None, X]
        assert X.ndim == 2

        risk = self.model.forward(X).cpu().detach().numpy().ravel()
        return self.risk_to_decision(risk), risk
    
    def set_feature_extractor(self, feature_extractor: FeatureExtractor):
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
            trained_epoch = f"trained{round(self.trained_epoch, -2)}"
        except AttributeError:
            trained_epoch = ""

        return f"{self.APPROACH}_{arch}_{out_assessment}_{xdim}_{patience}_{lr}"#_{trained_epoch}"

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



class ResNetRiskEstimator(MarkovianRiskEstimator):
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
        
        print(pred)
        self.model.train()
        return pred, pred #risk

class CombinedGPRiskEstimator():
    def __init__(self, *args, **kwargs):
        self.models = [
            GPRiskEstimator(*args, **kwargs),
            GPRiskEstimator(*args, **kwargs),
        ]
    
    def sample(self, input_data):
        
        def get_alphas_from_observations(data):
            # this assert might be deleted later
            # I need to check here that I get alpha (time) of demonstration
            assert len(data[0]) in [10,14,18,26,34,50,66]

            return data[:,-2]
        
        alphas = get_alphas_from_observations(input_data)

        r = []
        for alpha in alphas:
            # choose model
            r.append(self.models[int(alpha * len(self.models))].predict(input_data))

        return r

    def load_model(self):
        for n,model in enumerate(self.models):
            model.load_model(model_special=f"{n}")
    
    def save_model(self):
        for n,model in enumerate(self.models):
            model.save_model(model_special=f"{n}")

    def training_loop(self, dataloaders):
        assert isinstance(dataloaders, list), "There must be a list of dataloaders!"
        for dl,model in zip(dataloaders,self.models):
            print(f"training new dataloader")
            model.training_loop(dl)

class GPRiskEstimator(MarkovianRiskEstimator):
    APPROACH = "GP"

    def __init__(self,
                 name: str,
                 xdim: int = None, 
                 batch_size: int = 40, 
                 thr: float = 0.5, 
                 ard: bool = True,
                 learning_rate: float = 0.01,
                 arch: str = "",
                 out_assessment: str = 'cautious',
                 train_patience: int = 3000,
                 train_epoch: int = 3000,
        ):
        super(GPRiskEstimator, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.thr = thr
        self.xdim = xdim
        self.ard = ard
        self.learning_rate = learning_rate
        self.arch = arch
        self.out_assessment = out_assessment
        print("out_assessment: ", self.out_assessment)
        self.patience = train_patience
        self.train_epoch = train_epoch

    def load_model(self, model_special:str = ""):
        print(f"Loading Risk Estimation model: {self.model_path}/{self.name}_{self.__class__.__name__}_model{model_special}.pt")
        checkpoint = torch.load(f"{self.model_path}/{self.name}_{self.__class__.__name__}_model{model_special}.pt")

        self.create_model(checkpoint['X'], checkpoint['Y'])

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.move_model_to_cuda()

    def save_model(self, model_special:str = ""):
        """Overloaded function, saves also ra_model
        """        
        pathlib.Path(f"{self.model_path}").mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(), 
            "X": self.model.train_x,
            "Y": self.model.train_y,
            }, f"{self.model_path}/{self.name}_{self.__class__.__name__}_model{model_special}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(), 
            "X": self.model.train_x,
            "Y": self.model.train_y,
            }, f"{self.model_path}/{self.name}_{self.encode_params_as_str()}_model{model_special}.pt")

        
    def create_model(self, X, Y):        

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        print("Has  analytical likelihood:")
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

        # print("b4 ", self.model.covar_module.base_kernel.lengthscale)
        # self.model.covar_module.base_kernel.lengthscale = 10.0
        # print("af ", self.model.covar_module.base_kernel.lengthscale)


    def move_model_to_cuda(self):
        self.model=self.model.cuda()
        self.likelihood=self.likelihood.cuda()



    def training_loop(self, dataloader, early_stop=False):
        """_summary_

        Args:
            dataloader (_type_): _description_
            early_stop (bool, optional): Slowing down training! Defaults to False.
        """        

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

        early_stopping = GPEarlyStoppingAndPlot(self.patience, dataloader, validation_dataloader, self.dataloader_test_for_plot, self.dataloader_nodrop_for_plot)
        epochs_iter = tqdm(range(self.train_epoch))
        try:
            for i in epochs_iter:
    
                optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, Y)
                loss.backward()
                optimizer.step()
                
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
        print("Training finished")
        # print("Lengthscale: ")
        # print(self.model.covar_module.base_kernel.lengthscale)
        # print("Outputscale: ")
        # print(torch.sqrt(self.model.covar_module.outputscale))
    
    def sample(self, X):
        if X.ndim == 1:
            X = X[None, X]
        assert X.ndim == 2

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            mean = observed_pred.mean.cpu().numpy()
            std = observed_pred.stddev.cpu().numpy()
            reconstr_err = X[:,-1].cpu().numpy()
            
            if self.out_assessment == 'optimistic':
                risk = mean
            elif self.out_assessment == 'cautious':
                risk = mean + std
            elif self.out_assessment == 'max':
                risk = np.max(np.array([mean, std]), axis=0)
            else: raise Exception()

            # print(f"observed variance: {std}")
            pred = self.risk_to_decision(risk)
        
        self.model.train()
        self.likelihood.train()
        return pred, risk
    
    def sample_uncertainty(self, X):
        if X.ndim == 1:
            X = X[None, X]
        assert X.ndim == 2

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            mean = observed_pred.mean.cpu().numpy()
            std = observed_pred.stddev.cpu().numpy()

            std_pred = self.risk_to_decision(std)
        
        self.model.train()
        self.likelihood.train()
        return std_pred, std

import collections
import matplotlib.pyplot as plt

class GPEarlyStoppingAndPlot():
    def __init__(self, 
            patience: int = 120, # stopping patience in epochs
            # All dataloader used for plotting the train accuracy figure
            dataloader=None,
            validation_dataloader=None,
            test_dataloader=None, 
            nodrop_dataloader=None,
            use_test_data_for_stopping: bool = True, # For testing purposes
        ):
        self.patience = patience
        self.all_acc_trains = []
        self.all_acc_validations = []
        self.all_acc_tests = []
        self.all_acc_alldrops = []
        self.acc_trains = collections.deque(maxlen=patience)
        self.acc_validations = collections.deque(maxlen=patience)
        self.acc_tests = collections.deque(maxlen=patience)

        self.prepare_validation_data(dataloader, validation_dataloader, test_dataloader, nodrop_dataloader)

        self.use_test_data_for_stopping = use_test_data_for_stopping

    def prepare_validation_data(self, dataloader, validation_dataloader, test_dataloader, nodrop_dataloader):
        self.X_train, Y_train = RiskEstimationDataset.dataloader_to_array(dataloader)
        self.Y_train = Y_train.cpu().numpy().squeeze()
        self.X_validation, Y_validation = RiskEstimationDataset.dataloader_to_array(validation_dataloader)
        self.Y_validation = Y_validation.cpu().numpy().squeeze()

        if test_dataloader is not None:
            self.X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dataloader)
            self.Y_test = Y_test.cpu().numpy().squeeze()
            self.X_nodrop, Y_nodrop = RiskEstimationDataset.dataloader_to_array(nodrop_dataloader)
            self.Y_nodrop = Y_nodrop.cpu().numpy().squeeze()

            
    def __call__(self, epoch, risk_estimator):
        acc_train, acc_validation, acc_test, acc_nodrop = self.validate(risk_estimator)

        self.acc_trains.append(acc_train)
        self.acc_validations.append(acc_validation)
        self.acc_tests.append(acc_test)
        
        self.all_acc_trains.append(acc_train)
        self.all_acc_validations.append(acc_validation)
        self.all_acc_tests.append(acc_test)
        self.all_acc_alldrops.append(acc_nodrop)

        if self.use_test_data_for_stopping:
            if (acc_test <= sum(self.acc_tests)/len(self.acc_tests) and epoch > self.patience) or \
                (acc_test > 99 and acc_train > 99) or (acc_train > 99 and acc_test > 96 and self.acc_tests[-2] > acc_test):
                print(f"Early stopping on epoch {epoch}, acc_train: {acc_train}")
                return True
            else:
                return False
        else:
            if (acc_validation <= sum(self.all_acc_validations)/len(self.acc_validations) and epoch > self.patience) or \
                (acc_validation > 99 and acc_train > 99) or (acc_train > 99 and acc_validation > 96 and self.acc_validations[-2] > acc_validation):
                print(f"Early stopping on epoch {epoch}, acc_train: {acc_train}")
                return True
            else:
                return False

    def validate(self, risk_estimator):

        Y_pred, _ = risk_estimator.sample(self.X_train)
        acc_train =  100 * (self.Y_train == Y_pred).mean()
        Y_pred, _ = risk_estimator.sample(self.X_validation)
        acc_validation =  100 * (self.Y_validation == Y_pred).mean()

        acc_test = None
        acc_nodrop = None
        if risk_estimator.dataloader_test_for_plot is not None:
            Y_pred, _ = risk_estimator.sample(self.X_test)
            acc_test =  100 * (self.Y_test == Y_pred).mean()
            Y_pred, _ = risk_estimator.sample(self.X_nodrop)
            acc_nodrop =  100 * (self.Y_nodrop == Y_pred).mean()
        
        return acc_train, acc_validation, acc_test, acc_nodrop

    def plot_save(self, skill_name, risk_estimator):
        plt.figure(1, figsize=(6, 6))
        plt.plot(np.array([self.all_acc_trains, self.all_acc_validations, self.all_acc_tests, self.all_acc_alldrops]).T, linewidth=2)
        plt.legend(["Train", "Validation", "Test", "All Drops"])
        # plt.show()
        path = f"{risk_estimation.path}/autogen/{get_session()}/{skill_name}/"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        plt.savefig(path+"trainplot_"+risk_estimator.encode_params_as_str()+".svg")
        plt.close()

class MLPRiskEstimator(MarkovianRiskEstimator):
    APPROACH = "MLP"

    def __init__(self, 
                 name: str,
                 xdim: int = 8,
                 batch_size: int = 40,
                 thr: float = 0.5,
                 arch: str = 'A4',
                 learning_rate: float = 0.0001,
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

    def get_classifier_arcitecture(self, xdim, arch):
        if arch == 'LR':
            return LRRiskClassifier(xdim)
        if arch == 'A1':
            return RiskClassifierA1(xdim)
        elif arch == 'A2':
            return RiskClassifierA2(xdim)
        elif arch == 'A3':
            return RiskClassifierA3(xdim)
        elif arch == 'A4':
            return RiskClassifierA4(xdim)
        elif arch == 'A5':
            return RiskClassifierA5(xdim)
        else: raise Exception()

    def training_loop(self, dataloader, early_stop=True):
        best_val_metric = np.inf
        found_new_best = False
        no_improvement_count = 0
        epochs_iter = tqdm(range(self.train_epoch))

        dataloader, validation_dataloader = self.split_dataloader(dataloader)

        early_stopping = GPEarlyStoppingAndPlot(self.patience, dataloader, validation_dataloader, self.dataloader_test_for_plot, self.dataloader_nodrop_for_plot)
        try:
            for i in epochs_iter:
                for inputs, labels in dataloader:
                    labels = torch.tensor(labels, dtype=torch.float32).cuda()
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs.squeeze(), labels.squeeze())
                    loss.backward()
                    self.optimizer.step()

                # Check for early stopping
                    if loss < best_val_metric:
                        best_val_metric = loss
                        found_new_best = True

                if early_stop:
                    if i%5 == 0:
                        if early_stopping(i, self):
                            break
                # if found_new_best:
                #     no_improvement_count = 0
                #     found_new_best = False
                # else:
                #     no_improvement_count += 1

                # if no_improvement_count >= self.patience:
                #     print(f"No improvement for {self.patience} epochs. Stopping training.")
                #     self.trained_epoch = i
                #     return
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


class DistanceRiskEstimator(MarkovianRiskEstimator):
    APPROACH = "DIST"

    def __init__(self, name: str, dist_fun = None, thr: float = None, batch_size = 40, video_embedder = None):
        self.name = name
        self.dist_fun = dist_fun
        self.thr = thr

        self.batch_size = batch_size
        
        if video_embedder is not None:
            self.load_representation(name, video_embedder)

    def risk_to_decision(self, prob: float) -> int:
        """3: Risk --> Decision

        Args:
            prob (float): Probability of riskiness
            thr (float, optional): Threshold of decision

        Returns:
            int: Decision (1 safe or 0 risk)
        """
        return np.array(prob > self.thr, dtype=int)
    

    def training_loop(self, dataloader):
        print("No training implemented")

    def load_model(self):
        """Load hyperparameters"""
        df = pd.read_csv(f"{self.model_path}/{self.name}_{self.__class__.__name__}_hyperparam.csv")
        _, dist_fun, thr = list(df.loc[0])[0:3]
        self.dist_fun = eval(dist_fun)
        self.thr = thr

        print(f"dist_fun: {self.dist_fun}, and thr: {self.thr} loaded")


    def save_model(self):
        """Save hyperparameters"""        
        df = pd.DataFrame(np.array([[self.dist_fun.__name__, self.thr]]), columns=['dist_fun', 'thr'])
        df.to_csv(f"{self.model_path}/{self.name}_{self.__class__.__name__}_hyperparam.csv", index_label='Time')

    def normalize_dist(self, dist):
        # TODO: Generalize
        if self.dist_fun == cosine:
            # cosine dist returns values range: 0-2
            # By dividing 2, then the dist range: 0-1
            return dist / 2
        else:
            return dist

    def sample(self, x):
        """ x: observations vector, where x[:,-1] is frame number """
        ldim = self.latent_dim

        try: # to cpu if on gpu
            x = x.cpu().numpy().squeeze()
        except:
            pass
        
        if x.ndim == 1:
            x = x[None, x]
        assert x.ndim == 2

        ret_pred = []
        ret_risk = []
        for x_ in x:
            
            latent = x_[0:ldim]
            frame_normnum = x_[ldim]
            
            l = len(self.latent_images)
            repr_frame_number = int(frame_normnum * l)

            z1 = self.latent_images[repr_frame_number]

            risk_dist = self.test(z1, z2=latent)
            risk_dist = self.normalize_dist(risk_dist)

            pred = self.risk_to_decision(risk_dist)
            
            ret_pred.append(pred)
            ret_risk.append(risk_dist)

        return np.array(ret_pred), np.array(ret_risk)

    def load_representation(self, name, video_embedder):
        data = RiskEstimationDataset.load_video_data(name)
        tensor_images = data[0]
        
        if SAVE_GPU_SPACE: # save some GPU memory by encoding through batches
            dl = DataLoader(tensor_images, batch_size=100)
            out = []
            with torch.no_grad():
                for batch in dl:
                    latent_images_batch = video_embedder.model.encoder(batch)
                    out.append(latent_images_batch)
            latent_images = torch.cat(out, dim=0)
        else:
            latent_images = video_embedder.model.encoder(tensor_images)
        
        self.latent_dim = video_embedder.latent_dim
        self.tensor_images = tensor_images.cpu().detach().numpy()
        self.latent_images = latent_images.cpu().detach().numpy()

    def latent_to_risk(self, latent_image, n_image: int):
        ''' 2 '''
        latent_test = latent_image.cpu().detach().numpy().ravel()
        latent_repr = self.latent_images[n_image].cpu().detach().numpy().ravel()
        return self.test(latent_test, latent_repr)
        
    def test(self, z1, z2):
        return self.dist_fun(z1, z2)

    @staticmethod
    def make_distance(t1, t2, dist_fun):
        dist = np.zeros((len(t1), len(t2)))
        for n in range(len(t1)):
            for m in range(len(t2)):
                dist[n,m] = dist_fun(t1[n], t2[m])
        return dist

    def encode_trajectory(self, name, video_embedder):
        video_embedder.load(name=name)
        encoded_traj_1 = video_embedder.model.encoder(video_embedder.tensor_images)
        return encoded_traj_1.cpu().detach().numpy()

    def test_all_on_video_names(self, video_names, video_embedder):
        """Loads video demonstrations, safe and dangerous videos
            Comparison is w.r.t. skill_video_name skill

        Args:
            skill_video_name (_type_): _description_
            safe_names (_type_): _description_
            video_embedder (_type_): _description_
        """
        Y_pred = []
        Y_test = []
        for video_name in video_names:
            dataset_traj = RiskEstimationDataset.load_dataset([video_name], video_embedder, frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
            features=LatentObservationsRiskLabels)
            X_traj = dataset_traj.X.cpu().detach().numpy()
            y_traj = dataset_traj.Y.cpu().detach().numpy()

            Y_pred_risk, test_path_idx = self.compare_trajectories(self.latent_images, X_traj)

            print("Risk mean: ", np.median(Y_pred_risk))
    
            y_pred = self.risk_to_decision(np.array(Y_pred_risk))
            Y_pred.extend(y_pred)

            y_traj = y_traj.squeeze()
            Y_test.extend((y_traj[test_path_idx]).squeeze())
        
        assert len(Y_test) == len(Y_pred)

        return np.array(Y_test), np.array(Y_pred)
    
    def compare_trajectories(self, encoded_traj_1: np.ndarray, encoded_traj_2: np.ndarray):
        pred = []
        idxs = []
        for n, (i1, i2) in enumerate(zip(encoded_traj_1, encoded_traj_2)):
            pred.append(self.test(i1, i2))
            idxs.append(n)
        assert len(pred) == len(idxs)
        return pred, idxs
    
    def cross_test(self, t1: Iterable[float], t2: Iterable[float]):
        """Calls self.dist_fun for every combination of t1 and t2

        Args:
            t1 (Iterable[float]): 
            t2 (Iterable[float]): 

        Returns:
            float[len(t1), len(t2)]: Distance 
        """        
        dist = np.zeros((len(t1), len(t2)))
        for n in range(len(t1)):
            for m in range(len(t2)):
                dist[n, m] = self.dist_fun(t1[n], t2[m])
        return dist

class LinSearchDistanceRiskEstimator(DistanceRiskEstimator):

    def find_optimal_threshold(self, X, Y_true, thresholds):
        best_threshold = None
        best_score = 0  # Assuming higher score is better; adjust based on metric
        
        for threshold in thresholds:
            self.thr = threshold
            Y_pred, _ = self.sample(X)
            score = accuracy_score(Y_true.cpu().numpy(), Y_pred)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    

    def training_loop(self, dataloader):
        """Searches for threshold and best-performing dist fun
        """
        self.dist_fun = cosine
        assert self.dist_fun == cosine

        X, Y_true = RiskEstimationDataset.dataloader_to_array(dataloader)
        Y_true = Y_true.squeeze()
        # cosine dist
        thresholds = np.linspace(0, 2, 50)  # 50 thresholds evenly spaced between 0 and 1
        cosine_optimal_threshold, cosine_optimal_score = self.find_optimal_threshold(X, Y_true, thresholds)

        print(f"Cosine dist")
        print(f"Optimal Threshold: {cosine_optimal_threshold}")
        print(f"Optimal Score: {cosine_optimal_score}")

        self.thr = cosine_optimal_threshold
        return

        # euclidean dist
        self.dist_fun = euclidean
        thresholds = np.linspace(0, 20, 50)  # 50 thresholds evenly spaced between 0 and 1
        euclidean_optimal_threshold, euclidean_optimal_score = self.find_optimal_threshold(X, Y_true, thresholds)

        print(f"Euclidean dist")
        print(f"Optimal Threshold: {euclidean_optimal_threshold}")
        print(f"Optimal Score: {euclidean_optimal_score}")

        if cosine_optimal_score > euclidean_optimal_score:
            print(f"Choosing Cosine")
            self.thr = cosine_optimal_threshold
            self.dist_fun = cosine
        else:
            print(f"Choosing Euclidean")
            self.thr = euclidean_optimal_threshold
            self.dist_fun = euclidean




class NMDistanceRiskEstimator(DistanceRiskEstimator):
    def sample(self, trajectories):
        """Non-Markovian sampler; Uses trajectories as batches

        Args:
            skill_video_name (_type_): _description_
            safe_names (_type_): _description_
            video_embedder (_type_): _description_
        """
        
        Y_risk = []
        Y_pred = []
        Y_idxs = []
        for trajectory in trajectories:
            y_risk, y_idxs = self.compare_trajectories(self.latent_images, trajectory)
            y_pred = self.risk_to_decision(np.array(y_risk))

            Y_risk.append(y_risk)
            Y_pred.append(y_pred)
            Y_idxs.append(y_idxs)
            
        return np.array(Y_pred, dtype=object), np.array(Y_risk, dtype=object), np.array(Y_idxs, dtype=object)
    



class MinHyperTrainDistanceRiskEstimator(DistanceRiskEstimator):
    def __init__(self, name: str, batch_size = 40, video_embedder = None):
        self.name = name
        self.batch_size = batch_size
        self.dist_fun = cosine
        self.thr = 0.5

        if video_embedder is not None:
            self.load_representation(name, video_embedder)

    def training_loop(self, dataloader, search_for_outliers: bool = True):
        """ Searches for threshold and best-performing dist fun
            Form of Gradient Descent
            Trains the logistic regression
        """

        X, y = RiskEstimationDataset.dataloader_to_array(dataloader)
        X = X.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()

        pred, risks = self.sample(X)
        
        if not search_for_outliers:
            self.thr = np.min(risks[y == 1]) - 1e-4
            print(f"Selected minimum threshold: {self.thr}")
            return

        else:
            outlier_thr = 0.02
            risky_array = risks[y == 1]
            max_outliers_tested = 5
            mins = []
            for i in range(max_outliers_tested):
                mins.append(np.min(risky_array))
                risky_array[np.argmin(risky_array)] = np.inf

                if i==0: continue # cannot compare: min[0] == min[-1]

                is_outlier = abs(mins[-1] - mins[0]) > outlier_thr
                if not is_outlier:
                    break
            
            print(f"Selected threshold, considering outliers: {mins[-1]}, mins {mins}")
            self.thr = mins[-1]
            plot_threshold_labelled(risks, y)

class NMMinHyperTrainDistanceRiskEstimator(NMDistanceRiskEstimator, MinHyperTrainDistanceRiskEstimator):
    pass


class LRHyperTrainDistanceRiskEstimator(DistanceRiskEstimator):
    def __init__(self, name: str, batch_size = 40, video_embedder = None):
        self.name = name
        self.batch_size = batch_size
        self.dist_fun = cosine
        
        if video_embedder is not None:
            self.load_representation(name, video_embedder)
    
    def compute_cost(self, X, y, y_pred):
        assert len(y) == len(y_pred)
        # Binary cross-entropy cost
        m = len(y)
        epsilon = 1e-5  # to prevent log(0)
        cost = -np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon)) / m
        return cost

    def sample_with_thr(self, x, thr):
        self.thr = thr
        ret, ret_risk = self.sample(x)
        return ret, ret_risk

    def compute_gradient_numerical(self, X, y, weight, eps=5e-2):
        # Numerical gradient approximation

        grad_plus,_ = self.sample_with_thr(X, weight + eps)
        grad_minus,_ = self.sample_with_thr(X, weight - eps)
        numerical_gradient = (self.compute_cost(X, y, grad_plus) - self.compute_cost(X, y, grad_minus)) / (2 * eps)
        return numerical_gradient

    def training_loop(self, dataloader):
        """ Searches for threshold and best-performing dist fun
            Form of Gradient Descent
            Trains the logistic regression
        """

        X, y = RiskEstimationDataset.dataloader_to_array(dataloader)
        X = X.cpu().numpy().squeeze()
        y = y.cpu().numpy().squeeze()
        
        threshold = 1.0
        self.learning_rate = 0.001
        cost_history = []

        last_grad = -1
        for i in range(self.train_epoch):
            
            gradient = self.compute_gradient_numerical(X, y, threshold)
            
            if abs(gradient) < 0.001: # getting out of local minimum
                threshold -= self.learning_rate * last_grad
            else:
                threshold -= self.learning_rate * gradient
                last_grad = gradient

            y_pred, _ = self.sample_with_thr(X, threshold)

            cost = np.sum(np.abs(y_pred - y))

            cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Acc: {(y_pred == y).mean()}")
                print(f"Gradient: {gradient}")
                print(f"Cost at iteration {i}: {cost}")
                print(f"Threshold {threshold}")

        self.thr = threshold
        print(f"Threshold is {self.thr}")


class NMLRHyperTrainDistanceRiskEstimator(NMDistanceRiskEstimator, LRHyperTrainDistanceRiskEstimator):
    pass


class NMDistanceRiskEstimatorDTW(NMDistanceRiskEstimator):
    # def __init__(self, name, dist_fun=None, thr=None, dtw_fun=cosine):
    #     super().__init__(name=name, dist_fun=dist_fun, thr=thr)
    #     self.dtw_fun = dtw_fun

    def compare_trajectories(self, encoded_traj_1, encoded_traj_2):
        distance, path = fastdtw(encoded_traj_1, encoded_traj_2, dist=cosine)
        pred = []
        for p1, p2 in path:
            pred.append(self.test(encoded_traj_1[p1], encoded_traj_2[p2]))
        assert len(pred) == len(np.array(path)[:,0])
        return pred, np.array(path)[:,1]
    

def interp(original_array, new_length):
    new_indices = np.linspace(0, len(original_array) - 1, new_length)
    return np.interp(new_indices, np.arange(len(original_array)), original_array)

def test_interp():
    original_array = np.linspace(0, 10, 400)
    interp_arr = interp(original_array, new_length=350)

    assert len(interp_arr) == 350


def get_image_triplet(
        video_embedder, 
        images_numpy, 
        saliency_intensity_factor:int=1000,
        include_saliency: bool = False, # Tested for False
        include_reconstruction_loss: bool = True, # Tested for True
        ): 
    
    images_new = np.zeros((len(images_numpy), 64, 64))
    for i in range(len(images_numpy)):
        img = cv2.resize(images_numpy[i], (64, 64))
        # img = image_corrector.correct_image(img)
        images_new[i] = img

    images_new = images_new[:, np.newaxis, :, :]
    images_cuda = torch.tensor(images_new, dtype=torch.float32).cuda()
    
    decoded_images = []
    saliency_images = []
    criterion = nn.MSELoss()
    cr = []
    for idx in range(len(images_numpy)):
        latent = video_embedder.model.encoder(images_cuda[idx:idx+1])
        decoded_img1 = video_embedder.model.decoder(latent)

        cr_ = criterion(decoded_img1, images_cuda[idx:idx+1])
        decoded_img = decoded_img1.detach().cpu().numpy().squeeze().astype(np.uint8)
        cr.append(cr_)

        decoded_images.append(decoded_img)
        # decoded_images.append(decoded_img.detach().cpu().numpy())
        if include_saliency:
            saliency_map = saliency_intensity_factor*get_saliency_map_for_image(video_embedder.model, decoded_img1.clone().detach()[0]).detach().cpu().numpy()
            raise Exception("Add saliency to the images")
        if include_reconstruction_loss:
            img_reconstr = np.zeros((16,64))
            cv2.putText(
                img_reconstr, 
                str(int(cr_)), 
                (0, 12), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0), 
                1, 
                2
            )
            img_reconstr = 255 - img_reconstr

        saliency_images.append(img_reconstr)
    
    print("Averaged reconstruction loss: ", float(sum(cr)/len(cr)))
    
    image_triplets = []
    for ori, dec, sal in zip(images_numpy, decoded_images, saliency_images):
        image_triplets.append(np.vstack((ori.squeeze(), dec.squeeze(), sal.squeeze())))

    image_triplets = np.array(image_triplets)

    return image_triplets

def video_triplets_save(video_name: str, video_embedder, risk_estimator, features, train_dataloader=None, folder="videos"):
    images_ = load_video(video_name)

    images = get_image_triplet(video_embedder, images_, saliency_intensity_factor=1000)

    video_name_without_trial = video_name.split("_trial_")[0]
    video_name_without_trial = video_name_without_trial.split("_test_")[0]

    # .. / video_skill / video_name with trial /
    path = f"{risk_estimation.path}/{folder}/{get_session()}/{video_name_without_trial}/{video_name}/"
    
    # make missing folders if missing
    Path(path).mkdir(parents=True, exist_ok=True)

    save_video(path, video_name, images, h=144, w=64)


def sample_and_save_on_video(video_name: str, video_embedder, risk_estimator, features, train_dataloader=None, folder="videos"):
    """ Test risk_estimator on sample video_name
        Save video (*.mp4) and risk series (*.csv) to folder (default: "/videos/")

    Args:
        video_name (str): Name
        video_embedder (_type_): _description_
        risk_estimator (_type_): _description_
        features (_type_): Feature Extraction
    """    
    if isinstance(features, str):
        features = eval(features)

    video_name_without_trial = video_name.split("_trial_")[0]
    video_name_without_trial = video_name_without_trial.split("_test_")[0]

    # .. / video_skill / video_name with trial /
    path = f"{risk_estimation.path}/{folder}/{get_session()}/{video_name_without_trial}/{video_name}/"
    
    # make missing folders if missing
    Path(path).mkdir(parents=True, exist_ok=True)


    dataset = RiskEstimationDataset.load_dataset([video_name], video_embedder,    
        frame_dropping_policy=NoFrameDroppingPolicy, # All frames are sampled 
        features=features
    )

    safe_labels = RiskEstimationDataset.load_dataset([video_name], video_embedder,    
        frame_dropping_policy=NoFrameDroppingPolicy, # All frames are sampled 
        features=LatentObservationsSafeLabels
    )
    
    pred, risks = risk_estimator.sample(dataset.X.squeeze())
    
    correct = (pred == dataset.Y.cpu().numpy().squeeze())
    safe_labels = safe_labels.Y.cpu().numpy().squeeze()
    risk_labels = dataset.Y.cpu().numpy().squeeze()

    if isinstance(train_dataloader, type(None)):
        has_label = np.zeros((len(correct)))
    else:
        try:
            has_label = train_dataloader.dataset.dataset.has_label # len 400
        except AttributeError:
            has_label = train_dataloader.dataset.has_label
        has_label = interp(has_label, len(correct)) # len adjusted to current video


    df = pd.DataFrame(np.array([risks, correct, safe_labels, risk_labels, has_label]).T, columns=['Risk', 'Correct', 'SafeTrue', 'RiskTrue', 'HasLabel'])
    df.to_csv(f"{path}/{video_name}_{risk_estimator.encode_params_as_str()}.csv", index_label='Time')
    


    save_models_index_list(path, video_name)

    save_video_index_list(f"{risk_estimation.path}/{folder}", parent_folder=folder)

