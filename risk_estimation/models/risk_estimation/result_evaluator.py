

import pathlib
from typing import Iterable
import pandas as pd
from risk_estimation.models.risk_estimation.saliency_map_generator import get_saliency_map_for_image
import numpy as np
import torch
from utils import visualize_labelled_video
from pretty_confusion_matrix import pp_matrix_from_data
from copy import deepcopy
import risk_estimation
from sklearn.metrics import f1_score

class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'

class ResultEvaluator():
    
    def __init__(self, 
                 iwanttosee: Iterable[str] = ["accuracy", "confusion_matrix", "images_where_wrong", "decoded_images_where_wrong", "image_triplets"], 
                 iwanttosave: Iterable[str] = ["accuracy", "confusion_matrix"],
                 savepath: str = f"{risk_estimation.path}/autogen/",
                 name: str = "Results"):
        """Features to be visualized

        Args:
            iwanttosee (Iterable[str], optional): _description_. Defaults to ["accuracy", ].
        """        
        self.iwanttosee = iwanttosee
        self.iwanttosave = iwanttosave
        self.name = name
        self.savepath = savepath

    def to_cpu(self, Y_test):
        Y_test = Y_test.cpu().numpy().squeeze()
        return Y_test

    def __call__(self, risk_estimator, video_embedder, X_test, Y_test, X_test_images=None, Y_test_images=None):
        Y_test = self.to_cpu(Y_test)
        Y_pred, _ = risk_estimator.sample(X_test)
        
        try:
            Y_pred_std, _ = risk_estimator.sample_uncertainty(X_test)
        except AttributeError:
            Y_pred_std = None

        print(f"Results: {self.name}")
        if "accuracy" in self.iwanttosee:
            self.acc(Y_test, Y_pred)
        if "accuracy" in self.iwanttosave:
            self.acc_save(Y_test, Y_pred)

        if "f1" in self.iwanttosee:
            self.f1(Y_test, Y_pred)

        if "confusion_matrix" in self.iwanttosee:
            self.confusion_matrix(Y_test, Y_pred)
        if "confusion_matrix" in self.iwanttosave:
            self.confusion_matrix_save(Y_test, Y_pred)

        if "images_where_wrong" in self.iwanttosee:
            self.images_where_wrong(Y_test, Y_pred, X_test_images, Y_test_images)
        if "images_where_wrong" in self.iwanttosave:
            raise NotImplementedError
        
        if "decoded_images_where_wrong" in self.iwanttosee:
            self.decoded_images_where_wrong(video_embedder, Y_test, Y_pred, X_test_images, Y_test_images)
        if "decoded_images_where_wrong" in self.iwanttosave:
            raise NotImplementedError

        if "image_triplets" in self.iwanttosee:
            self.image_triplets(video_embedder, Y_test, Y_pred, Y_pred_std, X_test_images, Y_test_images)
        if "image_triplets" in self.iwanttosave:
            raise NotImplementedError
        
        return self.return_wrong_samples(Y_test, Y_pred)
        

    def return_wrong_samples(self, Y_test, Y_pred):
        return f"{sum(Y_test != Y_pred)}/{len(Y_test)}"
        
    def acc(self, Y_test, Y_pred):
        print("======")
        print(f"Datasamples: {len(Y_test)}, Risky: {len(Y_test[Y_test==1])}, Safe: {len(Y_test[Y_test==0])}")
        acc = 100 * (Y_test == Y_pred).mean()
        print(f"Accuracy: {cc.OKGREEN}{self.savepath} {acc}{cc.E}")
        print("======")

    def acc_save(self, Y_test, Y_pred):
        acc = 100 * (Y_test == Y_pred).mean()
        f1_ = 100 * f1_score(Y_test, Y_pred)
        fp = sum(1 for true, pred in zip(Y_test, Y_pred) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(Y_test, Y_pred) if true == 1 and pred == 0)
        n_samples = len(Y_pred)
        pathlib.Path(self.savepath).mkdir(parents=True, exist_ok=True)
        savedata = {"Accuracy": acc, "F1 score": f1_, "FP": fp, "FN": fn, "n samples": n_samples, "RiskyFrames": len(Y_test[Y_test==1]), "SafeFrames": len(Y_test[Y_test==0])}
        df = pd.DataFrame(savedata, index=[0])
        df.to_csv(f"{self.savepath}/{self.name.replace(' ','_')}_acc.csv")

    def f1(self, Y_test, Y_pred):
        f1_ = 100 * f1_score(Y_test, Y_pred)
        print(f"F1: {f1_}")

    def confusion_matrix(self, Y_test, Y_pred):
        pp_matrix_from_data(Y_test, Y_pred, columns=["Safe", "Danger"], name=self.name, savepath=None) # "pip install ." inside downloaded repo https://github.com/petrvancjr/pretty-print-confusion-matrix

    def confusion_matrix_save(self, Y_test, Y_pred):
        pp_matrix_from_data(Y_test, Y_pred, columns=["Safe", "Danger"], name=self.name, savepath=self.savepath) # "pip install ." inside downloaded repo https://github.com/petrvancjr/pretty-print-confusion-matrix

    def images_where_wrong(self, Y_test, Y_pred, X_test_images, Y_test_images):
        indxs = np.where(Y_test != Y_pred)[0]
        
        x = X_test_images.cpu().numpy()[indxs] 
        y = Y_test_images.cpu().numpy()[indxs]

        labels = {}
        print(y)
        print(y.shape)
        if len(y) > 0:
            labels = {
                'risk_flag': y[:,0],
                'safe_flag': y[:,1],
            }
        
        visualize_labelled_video(x, labels, press_for_next_frame=True, printer=True)

    def decoded_images_where_wrong(self, video_embedder, Y_test, Y_pred, X_test_images, Y_test_images):
        indxs = np.where(Y_test != Y_pred)[0]

        decoded_images = []
        for idx in indxs:
            latent = video_embedder.model.encoder(X_test_images[idx:idx+1])
            decoded_img = video_embedder.model.decoder(latent)
            decoded_images.append(decoded_img.detach().cpu().numpy())
            
        print("now decoded images")
        x = decoded_images
        y = Y_test_images.cpu().numpy()[indxs]

        labels = {}
        print(y)
        print(y.shape)
        if len(y) > 0:
            labels = {
                'risk_flag': y[:,0],
                'safe_flag': y[:,1],
            }
        
        visualize_labelled_video(x, labels, press_for_next_frame=True, printer=True)

    def image_triplets(self, video_embedder, Y_test, Y_pred, Y_pred_std, X_test_images, Y_test_images,
                       saliency_intensity_factor = 1000,
                       ):
        indxs = np.where(Y_test != Y_pred)[0]
        
        dataset_images = X_test_images.cpu().numpy()[indxs] 
        y = Y_test_images.cpu().numpy()[indxs]

        decoded_images = []
        saliency_images = []
        for idx in indxs:
            latent = video_embedder.model.encoder(X_test_images[idx:idx+1])
            decoded_img = video_embedder.model.decoder(latent)
            decoded_images.append(decoded_img.detach().cpu().numpy())

            saliency_images.append(saliency_intensity_factor*get_saliency_map_for_image(video_embedder.model, decoded_img.clone().detach()[0]).detach().cpu().numpy())
            
        labels = {}
        if len(y) > 0:
            labels = {
                'risk_flag': y[:,0],
                'safe_flag': y[:,1],
            }
        if Y_pred_std is not None:
            labels['novelty_flag'] = Y_pred_std

        image_triplets = []
        for ori, dec, sal in zip(dataset_images, decoded_images, saliency_images):
            image_triplets.append(np.vstack((ori.squeeze(), dec.squeeze(), sal.squeeze())))


        visualize_labelled_video(image_triplets, labels, press_for_next_frame=True, printer=True, h=64*3, w=64)

        