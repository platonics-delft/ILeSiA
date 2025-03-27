from copy import deepcopy
import threading
from error_recovery.models.recovery_state_finder import RecoveryStateFinder
import numpy as np
from torch.utils.data import DataLoader
from typing import Iterable

import video_embedding, risk_estimation
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
from risk_estimation.models.risk_estimation.risk_feature_extractor import *
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.frame_dropping import *
from risk_estimation.models.risk_estimator import MLPRiskEstimator, GPRiskEstimator
from video_embedding.utils import all_trial_names, visualize_labelled_video, visualize_labelled_video_frame, get_session
from video_embedding.models.video_embedder import VideoEmbedder

from risk_estimation.models.risk_estimator import *
from video_embedding.models.nerual_networks.autoencoder import *

import rospy
import time
import threading

def check_estimator(func):
    def wrapper(self, *args, **kwargs):
        if self.video_embedder is None or self.risk_estimator is None:
            print("Estimator is not set.")
            return None
        return func(self, *args, **kwargs)
    return wrapper


class SafetyLayer:
    """Deployed model usage - most effective risk model used"""
    def __init__(
        self,
        skill_name: str,
        latent_dim: int = 12,
        feature_extractor=StampedLatentObservationsRiskLabels,
        frame_dropping_policy = OnlyLabelledFramesDroppingPolicy,
        model_not_found_is_ok: bool = False,
        out_assessment: str = "cautious",
        enable_train: bool = False,
        enable_risk_estimator: bool = True,
    ):
        """_summary_

        Args:
            skill_name (str, optional): _description_. Defaults to "peg_door".
            ingeneralshouldbe (_type_, optional): _description_. Defaults to 16.
            feature_extractor (_type_, optional): _description_.
            frame_dropping_policy (FrameDropper, optional): _description_. Defaults to OnlyLabelledFramesDroppingPolicy.
        """        
        self.frame_dropping_policy = frame_dropping_policy
        self.feature_extractor = feature_extractor
        self.out_assessment = out_assessment
        
        if latent_dim == 0:
            session = get_session()
            if '12' in session:
                latent_dim = 12
            elif '16' in session:
                latent_dim = 16
            elif '24' in session:
                latent_dim = 24
            elif '32' in session:
                latent_dim = 32
            else:
                raise Exception("Not found")

        self.video_embedder = VideoEmbedder(name=skill_name, latent_dim=latent_dim, nn_model=LargeAutoencoder)
        if not enable_risk_estimator:
            self.video_embedder = None
            return
        else:
            if model_not_found_is_ok:
                try:
                    self.video_embedder.load_model()
                except FileNotFoundError:
                    print("Loading model: The model not found!")
                    self.video_embedder = None
                    return
            else:
                self.video_embedder.load_model()

        # self.update_video_embedding(epoch=50)
        
        if enable_train:
            self.update()
        else:
            self.load()

        self.recovery_state_finder = RecoveryStateFinder()

        self.observations = None
        self.risk = 0.0
        self.workersem = threading.Semaphore()
        self.worker = threading.Thread(target=self.risk_estimator_thread)
        self.worker.start()

    @check_estimator
    def sample(self, o):
        return self.risk_estimator.sample(o)

    def get_sample_dataloader(self):
        ''' Manually specify frame on which to train
        Args:
            FrameDroppingPolicy: OnlyLabelledFramesDroppingPolicyRisk{skill}{1|2}
            
        '''
        self.video_names = all_trial_names(self.video_embedder.name, include_repr=True)
        print(f"training on: {self.video_names}")

        self.dataloader, self.test_dataloader = RiskEstimationDataset.load(
            video_names=self.video_names,
            video_embedder=self.video_embedder,
            batch_size=self.video_embedder.batch_size,
            frame_dropping_policy=eval(f"OnlyLabelledFramesDroppingPolicyRiskPegDoor"),#self.frame_dropping_policy,
            features=self.feature_extractor,
        )
        self.dataloader_images, self.test_dataloader_images = RiskEstimationDataset.load(
            video_names=self.video_names,
            video_embedder=self.video_embedder,
            batch_size=self.video_embedder.batch_size,
            frame_dropping_policy=eval(f"OnlyLabelledFramesDroppingPolicyRiskPegDoor"),#self.frame_dropping_policy,
            features=VideoObservationsRiskAndSafeLabels,
        )

        return self.dataloader
    
    def update_video_embedding(self, epoch: int):
        """Should be function with no parameters, as specific as possible.
        video_embedder model is loaded.

        Args:
            epoch (int): _description_. 
        """ 

        print(self.video_embedder.model_train_record)
        
        # Get train_names - model was trained these 
        train_names = []       
        
        for train_inst in self.video_embedder.model_train_record:
            train_names.extend(train_inst['train_names'])
        train_names = list(set(train_names))
        print("Video embedder was trained on", train_names)

        # Get all trajectory demonstrations found for this skill
        all_names = all_trial_names(self.video_embedder.name)
        print("Video embedder sees videos: ", all_names)

        # Get all trajectory demonstrations that haven't been trained on
        update_names = []
        for name in all_names:
            if name not in train_names:
                update_names.append(name)

        print("Video embedder will be updated on videos: ", update_names)

        self.video_embedder.frame_dropping = True
        self.video_embedder.name = all_names[0]
        self.video_embedder.load(all_names)
        new_lr = 0.3 # speed up the learning
        for param_group in self.video_embedder.optimizer.param_groups:
            param_group['lr'] = new_lr
        self.video_embedder.create_video(epoch=epoch)
        
        # self.video_embedder.save_model()
        # self.video_embedder.save_latent_trajectory()


    def load(self):
        self.risk_estimator = get_risk_estimator(
            approach="deploy",
            skill_name=self.video_embedder.name,
            xdim=self.feature_extractor.xdim(self.video_embedder.latent_dim),
            video_embedder=self.video_embedder,
            out_assessment=self.out_assessment,
        )
        self.risk_estimator.load_model()


    def update(self):
        """Update model on all found trial for specific skill_name
        """
        dataloader = self.get_sample_dataloader()
        
        self.risk_estimator = get_risk_estimator(
            approach="deploy",
            skill_name=self.video_embedder.name,
            xdim=self.feature_extractor.xdim(self.video_embedder.latent_dim),
            video_embedder=self.video_embedder,
            out_assessment=self.out_assessment,
        )
        
        self.risk_estimator.training_loop(dataloader)
        self.risk_estimator.save_model()

        X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(self.test_dataloader_images)
        # X_train_images, Y_train_images = RiskEstimationDataset.dataloader_to_array(self.dataloader_images)

        X_test, Y_test = RiskEstimationDataset.dataloader_to_array(self.test_dataloader)
        e = ResultEvaluator(name="Evaluating on test dataset", iwanttosee=["accuracy"], iwanttosave=[])
        e(self.risk_estimator, self.video_embedder, X_test, Y_test, X_test_images, Y_test_images)

        # X_train, Y_train = RiskEstimationDataset.dataloader_to_array(self.dataloader)
        # e = ResultEvaluator(name="Evaluating on train dataset", iwanttosee=["accuracy"], iwanttosave=[])
        # e(self.risk_estimator, self.video_embedder, X_train, Y_train, X_train_images, Y_train_images)

        # dataset = RiskEstimationDataset.load_dataset(self.video_names, self.video_embedder,
        #     frame_dropping_policy=NoFrameDroppingPolicy,
        #     features=self.feature_extractor,)
        # dataset_images = RiskEstimationDataset.load_dataset(self.video_names, self.video_embedder,
        #     frame_dropping_policy=NoFrameDroppingPolicy,
        #     features=VideoObservationsRiskAndSafeLabels,)
        # e = ResultEvaluator(name="Evaluating on NoDrop policy dataset", iwanttosee=["accuracy"], iwanttosave=[])
        # e(self.risk_estimator, self.video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)

    @check_estimator
    def get_estimated_risk(self, observations) -> bool:
        self.workersem.acquire()
        self.observations = observations
        self.workersem.release()
        return self.risk, self.risk_vall

    def risk_estimator_thread(self):
        while not rospy.is_shutdown():
            self.workersem.acquire()
            observations = deepcopy(self.observations)
            self.workersem.release()
            risk, risk_vall = self.estimate_risk(observations)    
            self.workersem.acquire()
            self.risk = risk
            self.risk_vall = risk_vall
            self.workersem.release()
            time.sleep(0.01)

    @check_estimator
    def estimate_risk(self, observations) -> int: # 0 safe, 1 risk
        if observations is None:
            return 0.0, 0.0
        x, _ = self.feature_extractor.extract(observations, self.video_embedder)
        t1 = time.perf_counter()
        system_risk_pred, risk, std = self.risk_estimator.sample(x)
        
        system_risk_pred = int(np.array(system_risk_pred).squeeze())
        risk = float(np.array(risk).squeeze())
        print(f"pred: {system_risk_pred}, risk: {risk}, std: {std}, alpha: {float(observations[4].squeeze())}, {time.perf_counter()-t1}")

        return system_risk_pred, risk
    

    ''' Some convenience addons '''
    def get_random_image(self):
        try:
            self.dataloader_images
            
        except AttributeError:
            self.get_sample_dataloader(frame_dropping_policy=OnlyLabelledFramesDroppingPolicy)
            
        try:
            X = self.dataloader_images.dataset.X
        except AttributeError:
            X = self.dataloader_images.dataset.dataset.X

        l = len(X)
        rand_n = np.random.randint(0,l)

        return X[rand_n]
    



def get_risk_estimator_from_args(args, video_embedder):
    """
        DistanceRiskEstimators need video_embedder to load repre. demontration
        Trained RiskEstimators need input dim. to initialize model
    """    
    features = eval(args.features)
    xdim = features.xdim(args.video_latent_dim)

    return get_risk_estimator(args.approach, args.skill_name, xdim, video_embedder, args.out_assessment, args.train_patience, args.train_epoch)

def get_risk_estimator(approach, skill_name, xdim, video_embedder, out_assessment="optimistic", train_patience=15000, train_epoch=4000):
    """
        DistanceRiskEstimators need video_embedder to load repre. demontration
        Trained RiskEstimators need input dim. to initialize model
    """    
    if approach == "deploy":
        # return GPRiskEstimator(name=skill_name, xdim=xdim, learning_rate=0.01, arch="L+GP", out_assessment=out_assessment, train_patience=train_patience, train_epoch=train_epoch)
        # return MLPRiskEstimator(name=skill_name, xdim=xdim, train_patience=train_patience, train_epoch=500)
        return TwinGPRiskEstimator(
            skill_name,
            xdim,
            learning_rate=0.01,
            arch="", 
            train_epoch=train_epoch,
            train_patience=train_patience,
            out_assessment=out_assessment,
        )
    # Linear Regression - Baseline
    if approach == 'LR':
        return MLPRiskEstimator(name=skill_name, xdim=xdim, train_patience=train_patience, train_epoch=train_epoch, arch="LR")
    # Multilayer perceptron
    elif approach == 'MLP':
        return MLPRiskEstimator(name=skill_name, xdim=xdim, train_patience=train_patience, train_epoch=train_epoch)
    # Gaussian processes models
    elif approach == 'GP':
        return GPRiskEstimator(name=skill_name, xdim=xdim, learning_rate=0.01, 
            out_assessment=out_assessment, train_patience=train_patience, train_epoch=train_epoch)
    elif approach == "L+GP":
        return GPRiskEstimator(name=skill_name, xdim=xdim, learning_rate=0.01, arch="L+GP",
            out_assessment=out_assessment, train_patience=train_patience, train_epoch=train_epoch)
    elif approach == "L+GP+1SKIP":
        return GPRiskEstimator(name=skill_name, xdim=xdim, learning_rate=0.01, arch="L+GP+1SKIP",
            out_assessment=out_assessment, train_patience=train_patience, train_epoch=train_epoch)
    elif approach == "L+GP+2SKIP":
        return GPRiskEstimator(name=skill_name, xdim=xdim, learning_rate=0.01, arch="L+GP+2SKIP", 
            out_assessment=out_assessment, train_patience=train_patience, train_epoch=train_epoch)  
    elif approach == "TwinGP":
        return TwinGPRiskEstimator(name=skill_name, xdim=xdim, learning_rate=0.01, 
            out_assessment=out_assessment, train_patience=train_patience, train_epoch=train_epoch)  
    # Distance based models
    elif approach == 'DistLS':
        return LinSearchDistanceRiskEstimator(name=skill_name, video_embedder=video_embedder)  
    elif approach == 'DistLR':
        return LRHyperTrainDistanceRiskEstimator(name=skill_name, video_embedder=video_embedder)  
    elif approach == 'DistMin':
        return MinHyperTrainDistanceRiskEstimator(name=skill_name, video_embedder=video_embedder)

    elif approach == 'resnet50':
        return ResNetRiskEstimator(name=skill_name, train_epoch=50) # number epochs overwriten here for this method

    else: raise Exception()
