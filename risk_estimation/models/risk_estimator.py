
import pathlib
from typing import Any, Dict, Iterable

from risk_estimation.models.risk_estimation.saliency_map_generator import get_saliency_map_for_image
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
import pandas as pd
from sklearn.metrics import accuracy_score
import risk_estimation
from risk_estimation.models.gaussian_process.gp_classifier import GPModel
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_classifier import LRRiskClassifier, RiskClassifierA4, BinaryClassifier
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import StampedLatentObservationsRiskLabels, LatentObservationsRiskLabels, LatentObservationsSafeLabels, FeatureExtractor, StampedDistLatentObservationsRiskLabels
from video_embedding.models.video_embedder import VideoEmbedder
from risk_estimation.plot_utils import plot_threshold_labelled
from video_embedding.utils import get_session, load, save_models_index_list, save_video, save_video_index_list, tensor_image_to_cv2

from video_embedding.image_processing import saved_img_processing

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

class RiskEstimatorBase():
    def __init__(self):
        super(RiskEstimatorBase, self).__init__()
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

    def sample(self, 
               X: torch.Tensor # 1D or 2D tensor
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise Exception()
        # if X.ndim == 1:
        #     X = X[None, X]
        # assert X.ndim == 2

        # risk = self.model.forward(X).cpu().detach().numpy().ravel()
        # return self.risk_to_decision(risk), risk, np.zeros(len(risk))
    
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


class TwinGPRiskEstimator():
    APPROACH = "TwinGP"
    TIME_FEATURE_INDEX = -1 # the last feature is time

    def __init__(self, *args, **kwargs):
        self.models = [
            GPRiskEstimator(*args, **kwargs),
            GPRiskEstimator(*args, **kwargs),
        ]

    @property
    def dataloader_test_for_plot(self):
        return self.models[0].dataloader_test_for_plot
    
    @dataloader_test_for_plot.setter
    def dataloader_test_for_plot(self, dataloader):
        dataloaders = self.split_dataloader_to_models(dataloader)
        for dataloader, model in zip(dataloaders, self.models):
            model.dataloader_test_for_plot = dataloader

    @property
    def dataloader_nodrop_for_plot(self):
        return self.models[0].dataloader_nodrop_for_plot

    @dataloader_nodrop_for_plot.setter
    def dataloader_nodrop_for_plot(self, dataloader):
        dataloaders = self.split_dataloader_to_models(dataloader)
        for dataloader, model in zip(dataloaders, self.models):
            model.dataloader_nodrop_for_plot = dataloader

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
        
        assert len(dataset[0][0]) in [9, 10, 13, 14, 17, 18]

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

        return [
            DataLoader(Subset(dataset, low_indices), batch_size=batch_size, shuffle=True),
            DataLoader(Subset(dataset, high_indices), batch_size=batch_size, shuffle=True),
        ] 

    def training_loop(self, dataloader, early_stop=True):
        
        dataloaders = self.split_dataloader_to_models(dataloader)
        
        for dl,model in zip(dataloaders,self.models):
            print(f"training new dataloader")
            model.training_loop(dl, early_stop=early_stop)

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
        include_reconstruction_loss: bool = True, # Tested for True
        ): 
    images_cuda = torch.tensor(images_numpy, dtype=torch.float32).cuda()
    
    decoded_images = []
    original_images = []
    loss_title_images = []
    criterion = nn.MSELoss()
    cr = []
    for idx in range(len(images_numpy)):
        original_image = tensor_image_to_cv2(images_cuda[idx:idx+1])

        decoded_img1 = video_embedder.model.forward_batched(images_cuda[idx:idx+1])
        cr_ = criterion(decoded_img1, images_cuda[idx:idx+1])

        decoded_img = tensor_image_to_cv2(decoded_img1)
        cr.append(cr_)

        decoded_images.append(decoded_img)

        if include_reconstruction_loss:
            loss_title_image = np.zeros((16,64))
            cv2.putText(
                loss_title_image, 
                str(round(float(cr_),5)), 
                (0, 12), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0), 
                1, 
                2
            )
            loss_title_image = 255 - loss_title_image

        original_images.append(original_image)
        loss_title_images.append(loss_title_image)
    
    print("Averaged reconstruction loss: ", float(sum(cr)/len(cr)))
    
    image_triplets = []
    for ori, dec, sal in zip(original_images, decoded_images, loss_title_images):
        image_triplets.append(np.vstack((ori.squeeze(), dec.squeeze(), sal.squeeze())))

    image_triplets = np.array(image_triplets)

    return image_triplets

def video_triplets_save(video_name: str, video_embedder, folder="videos"):
    data = load(video_name)
    
    images_ = saved_img_processing(data['img']).squeeze().unsqueeze(1).cpu().numpy() # to format [images, 1, w, h]

    images = get_image_triplet(video_embedder, images_)

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

    pred, risks, std = risk_estimator.sample(dataset.X.squeeze())

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


    df = pd.DataFrame(np.array([risks, correct, safe_labels, risk_labels, has_label, std]).T, columns=['Risk', 'Correct', 'SafeTrue', 'RiskTrue', 'HasLabel', 'Std'])
    df.to_csv(f"{path}/{video_name}_{risk_estimator.encode_params_as_str()}.csv", index_label='Time')

    save_models_index_list(path, video_name)

    save_video_index_list(f"{risk_estimation.path}/{folder}", parent_folder=folder)

