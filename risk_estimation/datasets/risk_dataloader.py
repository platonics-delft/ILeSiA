from typing import Iterable, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torchvision
from torchvision.transforms.functional import to_pil_image

from risk_estimation.datasets.frame_dropping import *
from risk_estimation.datasets.risk_feature_extractor import *
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import load
from video_embedding.image_processing import saved_img_processing


class RiskEstimationDataset(Dataset):
    def __init__(self, X, Y, imgs, batch_size: int = 40, transform=None, video_names=[]):
        """
        Args:
            X (numpy array or torch Tensor): The input features.
            Y (numpy array or torch Tensor): The labels.
            imgs (numpy array): Corresponding input image
            batch_size (int): Optional
            transform (Sequence of transformations): If it is set, probably image input
        """
        # convert X to torch Tensor
        self.X = torch.tensor(X, dtype=torch.float32).cuda()
        # convert Y to torch Tensor
        self.Y = torch.tensor(Y, dtype=torch.int).cuda()

        self.imgs = torch.tensor(imgs, dtype=torch.float32).cuda()
        self.video_names = video_names
        self.batch_size = batch_size
        self.transform = transform

    def to_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform is not None: # only for resnet
            x_pil = to_pil_image(self.X[idx])
            x_tf = self.transform(x_pil)
            x_tf = torch.tensor(256*(1-x_tf), dtype=torch.float32).cuda()
            return x_tf, self.Y[idx]
        else:
            return self.X[idx], self.Y[idx]

    @staticmethod
    def load_video_data(name: str):
        """ You give single video file name, it returns the extracted data
        Args:
            name (str): Name of demonstration (file)

        Returns:
            video_data (dict): Dictionary with keys:
            img: np.array, images [x, 1, 64, 64]
            risk_flag: np.array, risk flags [x,1]
            safe_flag: np.array, safe flags [x,1]
            novel_risk_flag: np.array, novel risk flags [x,1]
            novel_safe_flag: np.array, novel safe flags [x,1]
            frame_number: np.array, frame number [x,1]
        """
        data = load(file=name)
        
        video_data = {
            "frame_number": np.array([np.arange(len(data["img"])) / len(data["img"])]).T, 
            "risk_flag": np.zeros((1,len(data["img"]))).T,
            "safe_flag": np.zeros((1,len(data["img"]))).T,
            "novel_risk_flag": np.zeros((1,len(data["img"]))).T,
            "novel_safe_flag": np.zeros((1,len(data["img"]))).T,
        }
        # (num_images, h, w) -> (num_images, 1, h, w)
        video_data["img"] = saved_img_processing(data["img"]).squeeze().unsqueeze(1).cpu().numpy()
        if "risk_flag" in data: video_data["risk_flag"] = np.array([data["risk_flag"]]).T            
        if "safe_flag" in data: video_data["safe_flag"] = np.array([data["safe_flag"]]).T
        if "novel_risk_flag" in data: video_data["novel_risk_flag"] = np.array([data["novel_risk_flag"]]).T
        if "novel_safe_flag" in data: video_data["novel_safe_flag"] = np.array([data["novel_safe_flag"]]).T

        return video_data

    @staticmethod
    def dataloader_to_array(dataloader):
        X, Y = [], []
        for x, y in dataloader:
            X.append(x)
            Y.append(y)

        return torch.cat(X, dim=0), torch.cat(Y, dim=0)

    @classmethod
    def load_dataset(
        cls,
        video_names: Iterable[str],
        video_embedder,
        batch_size=40,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=LatentObservationsRiskLabels,
        transform=None,
        add_whiteblackimg=False,
    ):
        """ The main function to load the dataset from list of video names.

        Args:
            video_names (Iterable[str]): Video names used to collect the dataset
            video_embedder: Embeds videos into latent space
            frame_dropping_policy (cls, optional): Discrads samples.
            features (classmethod, optional): Look at risk_feature_extractor.py.

        Returns:
            RiskEstimationDataset: X,Y - features and labels you choose based on features class
        """        
        # 1. Frame drop
        video_data_list = []
        for name in video_names:
            data = frame_dropping_policy.filter_frames(cls.load_video_data(name))
            
            if len(data["img"]) == 0:
                continue  # no samples

            video_data_list.append(data)
        
        # 2. Extract features
        X, Y, imgs = [], [], []
        for data,video_name in zip(video_data_list, video_names):
            x, y = features.extract(data, video_embedder, video_name)

            X.append(x)
            Y.append(y)
            imgs.append(data["img"])

        if len(X) == 0:
            return RiskEstimationDataset([], [], [])

        if add_whiteblackimg: # Consider whole white and black images as risky
            ones = torch.ones((40,1,64,64)).cuda()
            frame_numbers = (0.025 * torch.arange(0,40)).unsqueeze(1).cuda()
            latent = video_embedder.model.encoder(ones)
            whites = torch.cat((latent, frame_numbers), axis=1)
            X.append(whites.cpu().detach().numpy())
            Y.append(np.ones((40,1,1)))
            imgs.append(ones.cpu().detach().numpy())
            
            zeros = torch.zeros((40,1,64,64)).cuda()
            frame_numbers = (0.025 * torch.arange(0,40)).unsqueeze(1).cuda()
            latent = video_embedder.model.encoder(zeros)
            blacks = torch.cat((latent, frame_numbers), axis=1)
            X.append(blacks.cpu().detach().numpy())
            Y.append(np.ones((40,1,1)))
            imgs.append(zeros.cpu().detach().numpy())
            
        return RiskEstimationDataset(np.vstack(X), np.vstack(Y), np.vstack(imgs), batch_size, transform=transform, video_names=video_names)

    @classmethod
    def load_dataloader(cls,
            video_names: Iterable[str],
            video_embedder,
            batch_size=40,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=LatentObservationsRiskLabels,
            transform=None,
            add_whiteblackimg=False,
        ):
        """ Wrapper to get DataLoader instead of dataset
        """
        return DataLoader(cls.load_dataset(video_names, video_embedder, batch_size, frame_dropping_policy, features, transform,add_whiteblackimg=add_whiteblackimg), batch_size=video_embedder.batch_size)

    
    resnet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    transform = None
