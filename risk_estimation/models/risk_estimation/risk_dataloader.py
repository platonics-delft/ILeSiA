from typing import Iterable, Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from risk_estimation.models.risk_estimation.frame_dropping import (
    NoFrameDroppingPolicy,
)

from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, VideoObservationsRiskLabels, VideoObservationsRiskAndSafeLabels
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import load

import risk_estimation.models.risk_estimation.image_corrector as image_corrector

class RiskEstimationDataset(Dataset):
    def __init__(self, X, Y, batch_size: int = 40, has_label=None, transform=None):
        """
        Args:
            X (numpy array or torch Tensor): The input features.
            Y (numpy array or torch Tensor): The labels.
            batch_size (int): Optional
            transform (Sequence of transformations): If it is set, probably image input
        """
        # convert X to torch Tensor
        self.X = torch.tensor(X, dtype=torch.float32).cuda()
        # convert Y to torch Tensor
        self.Y = torch.tensor(Y, dtype=torch.int).cuda()

        self.batch_size = batch_size
        self.has_label = has_label
        self.transform = transform

    def to_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=False)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform is not None:
            x_pil = to_pil_image(self.X[idx])
            x_tf = self.transform(x_pil)
            x_tf = torch.tensor(256*(1-x_tf), dtype=torch.float32).cuda()
            return x_tf, self.Y[idx]
        else:
            return self.X[idx], self.Y[idx]

    @staticmethod
    def load_video_data(name: str):
        """
        Args:
            name (str): Name of demonstration (file)

        Returns:
            torch.tensor: get tensor_images [x, 1, 64, 64]
            risk_flag: risk flags [x,1]
            safe_flag: safe flags [x,1]
            novelty_flag: novelty flags [x,1]
            frame_number: frame number arange [x,1]
            recovery_phase: [x,1]
        """
        data = load(file=name)
        images = data["img"]
        l = len(data["img"])
        assert len(data["risk_flag"].T) == len(
            data["img"]
        ), f"Lens for {name} are not matching {len(data['risk_flag'].T)} != {len(data['img'])}"
        try:
            risk_flag = data["risk_flag"]
        except KeyError:
            risk_flag = np.zeros((1,len(images)))  # Setting to zeroes
        try:
            safe_flag = data["safe_flag"]
        except KeyError:
            safe_flag = np.zeros((1,len(images)))  # Setting to zeroes
        try:
            novelty_flag = data["novelty_flag"]
        except KeyError:  # Loaded demonstration has no record of risk data
            novelty_flag = np.zeros((1,len(images)))
        try:
            recovery_phase = data["recovery_phase"]
        except KeyError:
            recovery_phase = -1.0 * np.ones((1,len(images)))

        images_new = np.zeros((len(images), 64, 64))

        # ADD IMAGE CORRECTOR HERE
        IMAGE_CORRECTOR_ENABLED = False
        if IMAGE_CORRECTOR_ENABLED and image_corrector.COLOR_MATCHING:
            # This is getting messy
            # I need to load reference image trajectory in order to do color matching
            name_reference = name.split("_trial_")[0] # extract 'peg_door' from 'peg_door_trial_0'
            data_reference = load(file=name_reference)
            images_reference = data_reference["img"]
            images_reference_new = np.zeros((len(images_reference), 64, 64))
            for i in range(len(images_reference_new)):
                images_reference_new[i] = cv2.resize(images_reference[i], (64, 64))

        for i in range(len(images)):
            img = cv2.resize(images[i], (64, 64))
            # ADD IMAGE CORRECTOR HERE
            if IMAGE_CORRECTOR_ENABLED:
                img = image_corrector.correct_image(img, images_reference_new[i].astype(np.uint8))
            images_new[i] = img

        images_new = images_new[:, np.newaxis, :, :]
        # Assuming 'images' is your NumPy array of shape (num_images, h, w)

        # Convert NumPy array to a PyTorch tensor
        tensor_images = torch.tensor(images_new, dtype=torch.float32)
        tensor_images = tensor_images.cuda()
        risk_flag = torch.tensor(np.array([risk_flag]).T, dtype=torch.int).cuda()
        safe_flag = torch.tensor(np.array([safe_flag]).T, dtype=torch.int).cuda()
        novelty_flag = torch.tensor(np.array([novelty_flag]).T, dtype=torch.int).cuda()
        recovery_phase = torch.tensor(np.array([recovery_phase]).T, dtype=torch.float32).cuda()

        frame_number = torch.tensor(np.array([np.arange(l) / l]).T, dtype=torch.float32).cuda()

        assert frame_number.shape[1] == 1 and len(frame_number.shape)==2, f"Shape: (X, 1) != {frame_number.shape}"
        assert risk_flag.shape[1] == 1 and len(frame_number.shape)==2, f"Shape: (X, 1) != {risk_flag.shape}"
        assert safe_flag.shape[1] == 1 and len(frame_number.shape)==2, f"Shape: (X, 1) != {safe_flag.shape}"
        assert novelty_flag.shape[1] == 1 and len(frame_number.shape)==2, f"Shape: (X, 1) != {novelty_flag.shape}"
        assert recovery_phase.shape[1] == 1, f"Shape: (X, 1) != {recovery_phase.shape}"

        return tensor_images, risk_flag, safe_flag, novelty_flag, frame_number, recovery_phase

    @classmethod
    def load_videos_data_dataloaders(
        cls, names: Iterable[str], batch_size: int, frame_dropping_policy
    ):
        """Set dataloaders (list) for multiple input video data
        Args:
            names (Iterable[str]): names of input videos to load
            batch_size (int)
            frame_dropping_policy (cls)

        Returns:
            Iterable[Dataloader]: Dataloader for each video name
        """
        dataloaders = []
        for name in names:
            dataset = TensorDataset(
                *frame_dropping_policy.filter_frames(cls.load_video_data(name))
            )

            if len(dataset) == 0:
                continue  # no samples

            dataloaders.append(
                DataLoader(dataset, batch_size=batch_size, shuffle=False)
            )
        return dataloaders

    @staticmethod
    def dataloader_extract_to_array(
        dataloader,
        features=VideoObservationsRiskLabels,
        video_embedder=None,
    ):
        X, Y = [], []

        for datasample in dataloader:
            x, y = features.extract(datasample, video_embedder)
            X.append(x)
            Y.append(y)

        return torch.cat(X, dim=0), torch.cat(Y, dim=0)

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
    ):
        """Loads full dataset.

        Args:
            video_names (Iterable[str]): Video names used to collect the dataset
            video_embedder (_type_): Needed to embed loaded videos
            batch_size (int, optional): Defaults to 40.
            frame_dropping_policy (cls, optional): Defaults to NoFrameDroppingPolicy.
            features (classmethod, optional): _description_. Defaults to LatentObservationsRiskLabels.

        Returns:
            RiskEstimationDataset: X, e.g. 10 * 40 samples, 8 features each
                                   Y, e.g. 10 * 40 labels
        """        
        X, Y = [], []
        dataloaders = cls.load_videos_data_dataloaders(
            video_names, batch_size, frame_dropping_policy
        )
        for dataloader,video_name in zip(dataloaders, video_names):
            for datasample in dataloader:
                x, y = features.extract(datasample, video_embedder, video_name)

                X.append(x.cpu().detach().numpy())
                Y.append(y.cpu().detach().numpy())

        if len(X) == 0:
            return RiskEstimationDataset([], [])

        has_label = RiskEstimationDataset.has_sample_mask(dataloaders)

        return RiskEstimationDataset(np.vstack(X), np.vstack(Y), batch_size, has_label=has_label, transform=transform)

    @classmethod
    def load(
        cls,
        video_names: Iterable[str],
        video_embedder: VideoEmbedder,
        batch_size: int = 40,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=LatentObservationsRiskLabels,
        transform=None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Loads Risk Estimator's split train and test datasets as dataloaders

        Args:
            video_names (Iterable[str]): Video names used to collect the dataset
            video_embedder (VideoEmbedder): Needed to embed loaded videos
            batch_size (int, optional): Defaults to 40.
            frame_dropping_policy (cls, optional): Defaults to NoFrameDroppingPolicy.
            features (classmethod, optional): Defaults to LatentObservationsRiskLabels.

        Returns:
            Tuple[DataLoader, DataLoader]: Train and test dataloaders
        """
        dataset = cls.load_dataset(video_names, video_embedder, batch_size, frame_dropping_policy, features, transform)

        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=0.2, random_state=42
        )

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        # Create DataLoader
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader

    @staticmethod
    def has_sample_mask(dataloaders, traj_len: int = 400):
        has_sample = np.zeros((traj_len), dtype=bool)
        
        for dataloader in dataloaders:
            for datasample in dataloader:
                framedata = datasample[4]
                for frame in framedata:
                    norm_frame = frame # time-frame
                    idx = int(norm_frame * traj_len)

                    has_sample[idx] = True

        return has_sample
    
    resnet_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    transform = None

    @classmethod
    def extended_load(cls, video_train_names, video_test_names, video_embedder, framedrop_policy, features, resnet_option=False):
        
        train_dataset = cls.load_dataset(video_train_names,
                                         video_embedder, 
                                         video_embedder.batch_size, 
                                         framedrop_policy, 
                                         features, 
                                         cls.transform
        )
        
        train_imgset = cls.load_dataset(video_train_names, video_embedder, video_embedder.batch_size, framedrop_policy, VideoObservationsRiskAndSafeLabels, cls.transform)
        
        if resnet_option:
            test_dataset = cls.load_dataset(video_test_names, video_embedder, video_embedder.batch_size, frame_dropping_policy=framedrop_policy, features=VideoObservationsRiskLabels,
            transform=cls.resnet_transform)
            test_imgset = cls.load_dataset(video_test_names, video_embedder, video_embedder.batch_size, frame_dropping_policy=framedrop_policy, features=VideoObservationsRiskAndSafeLabels,
            transform=cls.resnet_transform)
        else:
            test_dataset = cls.load_dataset(video_test_names, video_embedder, video_embedder.batch_size, framedrop_policy, features, cls.transform)
            test_imgset = cls.load_dataset(video_test_names, video_embedder, video_embedder.batch_size, framedrop_policy, VideoObservationsRiskAndSafeLabels, cls.transform)
        
        return train_dataset, train_imgset, test_dataset, test_imgset
    
        
