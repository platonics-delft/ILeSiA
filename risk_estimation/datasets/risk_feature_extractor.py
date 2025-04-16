
from typing import Tuple

import numpy as np
import torch, torchvision
from torch import nn
from scipy.spatial.distance import cosine
from torchvision.transforms.functional import to_pil_image

SAVE_GPU_MEMORY = False

class FeatureExtractor():
    pass

class VideoObservationsRiskLabels(FeatureExtractor):
    """ X = imgs, Y = Risk Labels are 1 """
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        X = data["img"]
        Y = data["risk_flag"]
        return X, Y
    
class LatentObservationsSafeLabels(FeatureExtractor):
    """ X = h, Y = Safe Labels are 1"""
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()
        if SAVE_GPU_MEMORY:
            X = video_embedder.model.encoder_batched(data["img"])
        else:
            X = video_embedder.model.encoder(data["img"])
        Y = data["safe_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

class VideoObservationsRiskAndSafeLabels(FeatureExtractor):
    """ X = imgs, Y = [Risk Labels, Safe Labels] """
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        X = data["img"]
        Y_list = []
        for r,s in zip(data["risk_flag"],data["safe_flag"]):
            Y_list.append((r,s))
        Y = torch.tensor(Y_list, dtype=torch.float32)
        return X, Y

class LatentObservationsRiskLabels(FeatureExtractor):
    """ X = h, Y = Risk Labels are 1 """
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()
        if SAVE_GPU_MEMORY:
            X = video_embedder.model.encoder_batched(data["img"])
        else:
            X = video_embedder.model.encoder(data["img"])
        Y = data["risk_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

class ResnetLatentObservationsRiskLabels(FeatureExtractor):
    """ X = h, Y = Risk Labels are 1 """
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        video_embedder.optimizer.zero_grad()

        imgs = data["img"]
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
        Y = data["risk_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n

    rgb_transform = resnet_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

class StampedLatentObservationsRiskLabels(FeatureExtractor):
    """ This is the feature extractor used in the paper
        X = [h, alpha], Y = Risk Labels are 1 """
    @classmethod
    def extract(cls, data, video_embedder, video_name=None):
        video_embedder.optimizer.zero_grad()
        latent = video_embedder.model.encoder(torch.tensor(data["img"], dtype=torch.float32).cuda()).detach().cpu().numpy()
        if False: # TEST_PLOT
            import matplotlib.pyplot as plt
            plt.hist(latent.detach().cpu().numpy(), bins=100)
            plt.show()
            plt.imshow(video_embedder.model(data["img"])[0,0].detach().cpu(),cmap="grey")
            plt.show()

        frame_numbers = data["frame_number"] # (x, 1, 1)

        # frame_numbers = frame_numbers.squeeze(2) # (x, 1) 
        
        X = np.hstack((latent, frame_numbers))
        # torch.cat((latent, frame_numbers), axis=1)
        # when data is in range -300 to 300 -> it is good to use the scaling
        # X = torch.cat((0.0015 * latent + 0.5, frame_numbers), axis=1)
        Y = data["risk_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 # latent dim + time

###
### DEPRECATED
###
class StampedDistLatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder, video_name=None):
        video_embedder.optimizer.zero_grad()
        latent = video_embedder.model.encoder(data["img"])

        frame_numbers = data["frame_number"] # (x, 1, 1)

        # ugly
        from risk_estimation.models.risk_estimator import DistanceRiskEstimator
        dre = DistanceRiskEstimator(video_embedder.name, dist_fun=cosine, thr=0.0, video_embedder=video_embedder)
        _, similarity_dist, _ = dre.sample(torch.cat((latent, frame_numbers), axis=1).detach().cpu().numpy()) # (x, 1, 1)

        similarity_dist = torch.tensor(np.array([similarity_dist]).T, dtype=torch.int).cuda()
        # frame_numbers = frame_numbers.squeeze(2) # (x, 1) 
        X = torch.cat((latent, similarity_dist, frame_numbers), axis=1)
        Y = data["risk_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 + 1 # latent dim + time + similarity dist

class StampedDistRecErrLatentObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder, video_name=None):
        video_embedder.optimizer.zero_grad()
        latent = video_embedder.model.encoder(data["img"])
        reconstructed_images = video_embedder.model.decoder(latent)
        
        criterion = nn.MSELoss()
        losses = []
        for rec_img, ori_img in zip(reconstructed_images, data["img"]):
            loss = criterion(rec_img, ori_img)
            losses.append([loss])
        frame_numbers = data["frame_number"] # (x, 1, 1)


        # ugly
        from risk_estimation.models.risk_estimator import DistanceRiskEstimator
        dre = DistanceRiskEstimator(video_embedder.name, dist_fun=cosine, thr=0.0, video_embedder=video_embedder)
        _, similarity_dist = dre.sample(torch.cat((latent, frame_numbers), axis=1).detach().cpu().numpy()) # (x, 1, 1)

        similarity_dist = torch.tensor(np.array([similarity_dist]).T, dtype=torch.int).cuda()
        
        losses = torch.tensor(losses).cuda()
        
        # frame_numbers = frame_numbers.squeeze(2) # (x, 1) 
        X = torch.cat((latent, similarity_dist, losses, frame_numbers), axis=1)
        Y = data["risk_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 + 1 + 1 # latent dim + time + similarity dist

class StampedVideoObservationsRiskLabels(FeatureExtractor):
    @classmethod
    def extract(cls, data: Tuple, video_embedder=None, video_name=None):
        X = torch.cat((data["img"], torch.tensor([data["frame_number"]])))
        Y = data["risk_flag"]
        return X, Y
    
    @staticmethod
    def xdim(n):
        return n + 1 # latent dim + time

