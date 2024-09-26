
from typing import Tuple

import numpy as np
import torch, torchvision
from torch import nn
from scipy.spatial.distance import cosine
from torchvision.transforms.functional import to_pil_image

SAVE_GPU_MEMORY = False

class FeatureExtractor():
    """Extracted from data Tuple, where:
    data[0]: Video embedded latent vector
    data[1]: Risk Flags
    data[2]: Safe Flags
    data[3]: Novelty Flags
    data[4]: Frame number (demonstraion image number)
    data[5]: Recovery phase (0-1)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, Y) (observations, labels)
    """    
    @staticmethod
    def RiskLabels(data):
        return data[1]

    @staticmethod
    def PhaseLabels(data):
        """ No label is zero """
        return 10 * data[5].clip(0,10)

class VideoObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        X = data[0]
        Y = cls.RiskLabels(data)
        return X, Y
    
    xreq = ['image']

class LatentObservationsSafeLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()
        if SAVE_GPU_MEMORY:
            X = video_embedder.model.encoder_batched(data[0])
        else:
            X = video_embedder.model.encoder(data[0])
        Y = data[2]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

    xreq = ['image']

class VideoObservationsRiskAndSafeLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        X = data[0]
        Y_list = []
        for r,s in zip(data[1],data[2]):
            Y_list.append((r,s))
        Y = torch.tensor(Y_list, dtype=torch.float32)
        return X, Y

    xreq = ['image']

class LatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()
        if SAVE_GPU_MEMORY:
            X = video_embedder.model.encoder_batched(data[0])
        else:
            X = video_embedder.model.encoder(data[0])
        Y = cls.RiskLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

    xreq = ['image']


class ResnetLatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()

        imgs = data[0]
        imgs_new = []
        for img in imgs:
            x_pil = to_pil_image(img)
            x_ = cls.rgb_transform(x_pil)
            x_ = torch.tensor(256 * (1-x_), dtype=torch.float32).cuda()
            imgs_new.append(x_.unsqueeze(0))

        imgs_new = torch.cat(imgs_new)

        if SAVE_GPU_MEMORY:
            X = video_embedder.model.encoder_batched(imgs_new)
        else:
            X = video_embedder.model.encoder(imgs_new)
        Y = cls.RiskLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

    xreq = ['image']    

    rgb_transform = resnet_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class LatentObservationsRiskLabelsPriorRisk(LatentObservationsRiskLabels):
    @staticmethod
    def RiskLabels(data):
        return 1-data[2]

class StampedLatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder, video_name=None):
        video_embedder.optimizer.zero_grad()
        latent = video_embedder.model.encoder(data[0])

        frame_numbers = data[4] # (x, 1, 1)

        # frame_numbers = frame_numbers.squeeze(2) # (x, 1) 
        X = torch.cat((latent, frame_numbers), axis=1)
        Y = cls.RiskLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 # latent dim + time

    xreq = ['image', 'frame_number']

class StampedLatentObservationsRiskLabelsPriorRisk(StampedLatentObservationsRiskLabels):
    @staticmethod
    def RiskLabels(data):
        return 1-data[2]

class StampedDistLatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder, video_name=None):
        video_embedder.optimizer.zero_grad()
        latent = video_embedder.model.encoder(data[0])

        frame_numbers = data[4] # (x, 1, 1)

        # ugly
        from risk_estimation.models.risk_estimator import DistanceRiskEstimator
        dre = DistanceRiskEstimator(video_embedder.name, dist_fun=cosine, thr=0.0, video_embedder=video_embedder)
        _, similarity_dist = dre.sample(torch.cat((latent, frame_numbers), axis=1).detach().cpu().numpy()) # (x, 1, 1)

        similarity_dist = torch.tensor(np.array([similarity_dist]).T, dtype=torch.int).cuda()
        # frame_numbers = frame_numbers.squeeze(2) # (x, 1) 
        X = torch.cat((latent, frame_numbers, similarity_dist), axis=1)
        Y = cls.RiskLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 + 1 # latent dim + time + similarity dist

    xreq = ['image', 'frame_number', 'similarity_dist']



class StampedDistRecErrLatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder, video_name=None):
        video_embedder.optimizer.zero_grad()
        latent = video_embedder.model.encoder(data[0])
        reconstructed_images = video_embedder.model.decoder(latent)
        
        criterion = nn.MSELoss()
        losses = []
        for rec_img, ori_img in zip(reconstructed_images, data[0]):
            loss = criterion(rec_img, ori_img)
            losses.append([loss])
        frame_numbers = data[4] # (x, 1, 1)


        # ugly
        from risk_estimation.models.risk_estimator import DistanceRiskEstimator
        dre = DistanceRiskEstimator(video_embedder.name, dist_fun=cosine, thr=0.0, video_embedder=video_embedder)
        _, similarity_dist = dre.sample(torch.cat((latent, frame_numbers), axis=1).detach().cpu().numpy()) # (x, 1, 1)

        similarity_dist = torch.tensor(np.array([similarity_dist]).T, dtype=torch.int).cuda()
        
        losses = torch.tensor(losses).cuda()
        
        # frame_numbers = frame_numbers.squeeze(2) # (x, 1) 
        X = torch.cat((latent, frame_numbers, similarity_dist, losses), axis=1)
        Y = cls.RiskLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 + 1 + 1 # latent dim + time + similarity dist

    xreq = ['image', 'frame_number', 'similarity_dist', 'losses']


class StampedDistLatentObservationsRiskLabelsPriorRisk(StampedDistLatentObservationsRiskLabels):
    @staticmethod
    def RiskLabels(data):
        return 1-data[2]


class StampedVideoObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        X = torch.cat((data[0], torch.tensor([data[3]])))
        Y = cls.RiskLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 # latent dim + time

    
    xreq = ['image', 'frame_number']



class LatentObservationsPhaseLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()
        X = video_embedder.model.encoder(data[0])
        Y = cls.PhaseLabels(data)
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

    xreq = ['image']