import gpytorch
import torch

from risk_estimation.models.risk_estimator import RiskEstimatorBase
import pathlib
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset
from tqdm import tqdm
from video_embedding.utils import get_session
import risk_estimation
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import deque 

# We will use the simplest form of GP model, exact inference
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard=False):
        self.train_x = train_x
        self.train_y = train_y
        if ard:
            ard_num_dim=train_x.size(-1)
        else:
            ard_num_dim=None
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()  
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dim,
                lengthscale_constraint=gpytorch.constraints.Interval(0.1, 1.0),
                ),
            # outputscale_constraint=gpytorch.constraints.Interval(0.5, 2.0)
        )

        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dim, lengthscale_constraint=gpytorch.constraints.Interval(0.1, 1.0))
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPRiskEstimator(RiskEstimatorBase):
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
            "X": self.model.train_x,
            "Y": self.model.train_y,
            }, f"{self.model_path}/{self.name}_{self.__class__.__name__}_model.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(), 
            "X": self.model.train_x,
            "Y": self.model.train_y,
            }, f"{self.model_path}/{self.name}_{self.encode_params_as_str()}_model.pt")

        
    def create_model(self, X, Y):        

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        print("Has  analytical likelihood:")
        print(self.likelihood.has_analytic_marginal)

        if self.arch == '':
            self.model = GPModel(X, Y, self.likelihood, ard=self.ard)
        else: raise Exception()

    def move_model_to_cuda(self):
        self.model=self.model.cuda()
        self.likelihood=self.likelihood.cuda()



    def training_loop(self, dataloader, early_stop:bool=False, see_all_losses:bool=False):
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

        if early_stop:
            early_stopping = GPEarlyStoppingAndPlot(self.patience, dataloader, validation_dataloader, self.validation_dataloaders, see_all_losses=see_all_losses)
        self.epochs_iter = tqdm(range(self.train_epoch))
        try:
            for i in self.epochs_iter:
    
                optimizer.zero_grad()
                output = self.model(X)
                loss = -mll(output, Y)

                if see_all_losses:
                    self.model.eval()
                    self.likelihood.eval()
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        valid_X = torch.vstack([i for i,j in validation_dataloader.dataset])
                        valid_Y = torch.vstack([j for i,j in validation_dataloader.dataset])
                        try:
                            valid_output = self.model(valid_X)
                            self.valid_loss = -mll(valid_output, valid_Y.squeeze()).item()
                        except: 
                            self.valid_loss = np.inf
                        test_X = torch.vstack([i for i,j in self.validation_dataloaders["test"].dataset])
                        test_Y = torch.vstack([j for i,j in self.validation_dataloaders["test"].dataset])
                        try:
                            valid_output = self.model(test_X)
                            self.test_loss = -mll(valid_output, test_Y.squeeze()).item()
                        except: 
                            self.test_loss = np.inf
                    self.model.train()
                    self.likelihood.train()

                loss.backward()
                self.loss = loss.item()
                optimizer.step()
                
                if early_stop:
                    if i%5 == 0:
                        if early_stopping(i, self):
                            break
        except KeyboardInterrupt:
            print("Stopping on interrupt")
        finally:
            print("Continuing with the rest of the program")
        
        if early_stop:
            early_stopping.plot_save(self.name, self)

        self.trained_epoch = i
        print("Training finished")
        # print("Lengthscale: ")
        # print(self.model.covar_module.base_kernel.lengthscale)
        # print("Outputscale: ")
        # print(torch.sqrt(self.model.covar_module.outputscale))
    
    def sample(self, X):
        """ Returns three numpy arrays """
        if X.ndim == 1:
            X = X.unsqueeze(0)
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
                pred = self.risk_to_decision(risk)
            elif self.out_assessment == 'cautious':
                risk = mean + std
                pred = self.risk_to_decision(risk)
            elif self.out_assessment == 'separate_thrs':
                risk = mean + std
                pred = self.custom_decision(mean, std)
            else: raise Exception()

        self.model.train()
        self.likelihood.train()
        return pred, risk, std

    def custom_decision(self, mean: float, std: float, thr_std: float = 0.3) -> int:
        """ Activation Logic """
        logic_pred = (np.array(std) > thr_std) or (std < thr_std and mean > self.thr)

        d_ = list(self.prev) + list(logic_pred)
        ret = []
        for i in range(self.WINDOW_SIZE+1, len(d_) + 1):
            window = d_[i - self.WINDOW_SIZE:i]
            ret.append(all(window))

        self.prev = deque(d_[-self.WINDOW_SIZE:], maxlen=self.WINDOW_SIZE)
    
        return np.array(ret)


class TwinGPRiskEstimator():
    APPROACH = "TwinGP"
    TIME_FEATURE_INDEX = -1 # the last feature is time

    def __init__(self, *args, **kwargs):
        self.models = [
            GPRiskEstimator(*args, **kwargs),
            GPRiskEstimator(*args, **kwargs),
        ]

    @property
    def validation_dataloaders(self):
        return self.models[0].validation_dataloaders

    def set_dataloaders_for_validation(self, list_of_dataloaders: list, names: list):
        for model in self.models: # empty
            model.validation_dataloaders = {}
        for dataloader, name in zip(list_of_dataloaders, names):
            dataloaders_model_split = self.split_dataloader_to_models(dataloader)
            if dataloaders_model_split is None: continue
            for dataloader, model in zip(dataloaders_model_split, self.models):
                model.validation_dataloaders[name] = dataloader

    def sample(self, 
               X: torch.Tensor, # 1D or 2D tensor
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # 1D, 1D, 1D

        def get_alphas_from_observations(data):
            assert len(data[0]) in [9, 10, 13, 14, 17, 18]
            return data[:,self.TIME_FEATURE_INDEX]
        
        alphas = get_alphas_from_observations(X).detach().cpu().numpy()
        
        model1_mask = alphas <= 0.5
        model2_mask = alphas > 0.5
        
        X1, X2 = X[model1_mask], X[model2_mask]

        if len(X1) > 0:
            preds1, risks1, stds1 = self.models[0].sample(X1)
        if len(X2) > 0:
            preds2, risks2, stds2 = self.models[1].sample(X2)

        # Prepare output arrays

        preds, risks, stds = np.empty(len(X)), np.empty(len(X)), np.empty(len(X))

        # Assign results back based on original mask
        if len(X1) > 0:
            preds[model1_mask], risks[model1_mask], stds[model1_mask] = preds1, risks1, stds1
        if len(X2) > 0:
            preds[model2_mask], risks[model2_mask], stds[model2_mask] = preds2, risks2, stds2

        return preds, risks, stds
    
    def encode_params_as_str(self):
        return self.models[0].encode_params_as_str()+"_twin"

    def load_model(self):

        print(f"Loading Risk Estimation model: {self.models[0].model_path}/{self.models[0].name}_{self.__class__.__name__}_model.pt")
        checkpoints = torch.load(f"{self.models[0].model_path}/{self.models[0].name}_{self.__class__.__name__}_model.pt")

        for checkpoint, model in zip(checkpoints, self.models):
            model.create_model(checkpoint['X'], checkpoint['Y'])
            model.model.load_state_dict(checkpoint['model_state_dict'])
            model.model.eval()
            model.move_model_to_cuda()

    def save_model(self):
        model_to_save = []
        for n,model in enumerate(self.models):
            model_to_save.append(
                {
                    "model_state_dict": model.model.state_dict(), 
                    "X": model.model.train_x,
                    "Y": model.model.train_y,
                }
            )
        pathlib.Path(f"{self.models[0].model_path}").mkdir(parents=True, exist_ok=True)
        torch.save(model_to_save, f"{self.models[0].model_path}/{self.models[0].name}_{self.__class__.__name__}_model.pt")
        torch.save(model_to_save, f"{self.models[0].model_path}/{self.models[0].name}_{self.models[0].encode_params_as_str()}_model.pt")

    def split_dataloader_to_models(self, dataloader):
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size
        

        low_indices = []
        high_indices = []
        
        for i in range(len(dataset)):
            x, y = dataset[i]  # Extract (x, y)
            
            if x[self.TIME_FEATURE_INDEX] < 0.4:
                low_indices.append(i)
            elif x[self.TIME_FEATURE_INDEX] < 0.6:
                low_indices.append(i)
                high_indices.append(i)
            else:
                high_indices.append(i)
        
        try:
            return [
                DataLoader(Subset(dataset, low_indices), batch_size=batch_size, shuffle=True),
                DataLoader(Subset(dataset, high_indices), batch_size=batch_size, shuffle=True),
            ] 
        except ValueError:
            print("Dataset is empty!")
            return None

    def training_loop(self, dataloader, early_stop=True):
        
        dataloaders = self.split_dataloader_to_models(dataloader)
        
        for dl,model in zip(dataloaders,self.models):
            print(f"training new dataloader")
            model.training_loop(dl, early_stop=early_stop)


import matplotlib.pyplot as plt

class GPEarlyStoppingAndPlot():
    def __init__(self, 
            patience: int = 120, # stopping patience in epochs
            # All dataloader used for plotting the train accuracy figure
            dataloader=None,
            validation_dataloader=None,
            validation_dataloaders={},
            use_test_data_for_stopping: bool = True, # For testing purposes
            see_all_losses: bool = False,
        ):
        self.patience = patience

        self.acc_plot_data = {"train": [], "valid": []}
        self.prepare_validation_data(dataloader, validation_dataloader, validation_dataloaders)

        self.use_test_data_for_stopping = use_test_data_for_stopping
        self.see_all_losses = see_all_losses

    def prepare_validation_data(self, dataloader, validation_dataloader, validation_dataloaders):
        self.X_train, Y_train = RiskEstimationDataset.dataloader_to_array(dataloader)
        self.Y_train = Y_train.cpu().numpy().squeeze()
        self.X_validation, Y_validation = RiskEstimationDataset.dataloader_to_array(validation_dataloader)
        self.Y_validation = Y_validation.cpu().numpy().squeeze()

        self.validation_datasets = {}

        for name in validation_dataloaders.keys():
            dataloader = validation_dataloaders[name]
            X, Y = RiskEstimationDataset.dataloader_to_array(dataloader)
            self.validation_datasets[name] = {}
            self.validation_datasets[name]["X"] = X
            self.validation_datasets[name]["Y"] = Y.cpu().numpy().squeeze()
            
            self.all_accs_validations = {}
            self.acc_plot_data[name] = []

    def __call__(self, epoch, risk_estimator):
        acc_train, acc_validation, accs = self.validate(risk_estimator)

        self.acc_plot_data["train"].append(acc_train)
        self.acc_plot_data["valid"].append(acc_validation)
        for name in self.validation_datasets.keys():
            self.acc_plot_data[name].append(accs[name])


        # risk_estimator.epochs_iter.set_description(f"Tr: {acc_train:3.0f}%, Test: {accs['test']:3.0f}%, loss: {risk_estimator.loss}, Lengthscale grad: {risk_estimator.model.covar_module.base_kernel.lengthscale.grad} Out scale grad: {risk_estimator.model.covar_module.outputscale.grad}")
        if self.see_all_losses:
            risk_estimator.epochs_iter.set_description(f"Tr: {acc_train:3.0f}%, Test: {accs['test']:3.0f}%, loss: {risk_estimator.loss:.3f}, valid loss: {risk_estimator.valid_loss:.3f}, test loss: {risk_estimator.test_loss:.3f}")
            print(list(risk_estimator.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().squeeze()))
        else:
            risk_estimator.epochs_iter.set_description(f"Tr: {acc_train:3.0f}%, Test: {accs['test']:3.0f}%, loss: {risk_estimator.loss:.3f}")

        if (acc_validation <= sum(self.acc_plot_data["valid"])/len(self.acc_plot_data["valid"][-10:]) and epoch > self.patience):
            print(f"Early stopping on epoch {epoch}, acc_train: {acc_train}")
            return True
        else:
            return False

    def validate(self, risk_estimator):

        Y_pred, _, _ = risk_estimator.sample(self.X_train)
        acc_train =  100 * (self.Y_train == Y_pred).mean()
        Y_pred, _, _ = risk_estimator.sample(self.X_validation)
        acc_validation =  100 * (self.Y_validation == Y_pred).mean()

        accs = {}
        for name in self.validation_datasets.keys():
            dataset = self.validation_datasets[name]
            Y_pred, _, _ = risk_estimator.sample(dataset["X"])
            accs[name] = 100 * (dataset["Y"] == Y_pred).mean()
        
        return acc_train, acc_validation, accs

    def plot_save(self, skill_name, risk_estimator):
        plt.figure(1, figsize=(6, 6))
        
        plt.plot(np.array(list(self.acc_plot_data.values())).T, linewidth=2)
        plt.legend(list(self.acc_plot_data.keys()))
        # plt.show()
        path = f"{risk_estimation.path}/autogen/{get_session()}/{skill_name}/"
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        plt.savefig(path+"trainplot_"+risk_estimator.encode_params_as_str()+".svg")
        plt.close()
