from copy import deepcopy
import os
import numpy as np
import rospkg
import risk_estimation
from risk_estimation.models.risk_estimation.frame_dropping import ProactiveRiskLabelingDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import StampedLatentObservationsRiskLabels
from risk_estimation.models.risk_estimator import MLPRiskEstimator
import video_embedding
from video_embedding.models.video_embedder import VideoEmbedder
from pretty_confusion_matrix import pp_matrix_from_data
from video_embedding.utils import number_of_saved_trials, set_session


def test_risk_flag_extremes():

    input_array =  np.array([0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,])
    output_array = np.array([0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,])

    ProactiveRiskLabelingDroppingPolicy.near_radius = 3
    extremes = ProactiveRiskLabelingDroppingPolicy.risk_flag_points_of_interest(input_array)
    compare = np.zeros(len(input_array), dtype=int)
    compare[extremes] = 1

    assert (output_array == compare).all(), f"{output_array}\n{compare}"


def prepare(skill_name):
    # 1. Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/", name=skill_name,
    )

    # 2. Risk estimator with hyper params
    risk_estimator = MLPRiskEstimator(
        skill_name,
        9,
        video_embedder.batch_size,
    )
    return video_embedder, risk_estimator

def data_aggregation(skill_name, video_names, plotter=False):
    video_embedder, risk_estimator = prepare(skill_name)


    n = number_of_saved_trials(skill_name)
    all_video_names = [f"{skill_name}_trial_{i}" for i in range(n)]
    # Load test dataset from all trials
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names=all_video_names,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
        features=StampedLatentObservationsRiskLabels,
    )
    all_trials_test_dataloader = deepcopy(test_dl)
    
    for n in range(len(video_names)):
        print(f"--- {n} ---")
        print(f"Video names: {video_names[:n+1]}")
        train_dl, test_dl =RiskEstimationDataset.load(
            video_names=video_names[:n+1],
            video_embedder=video_embedder,
            batch_size=video_embedder.batch_size,
            frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
            features=StampedLatentObservationsRiskLabels,
        )
        risk_estimator.training_loop(
            train_dl, num_epochs=1500, patience=1000, epoch_print=200
        )  # Train Risk Aware module
        X, Y = RiskEstimationDataset.dataloader_to_array(
            train_dl)
        print(X.shape)
        print(Y.shape)


        Y = Y.cpu().numpy().squeeze()
        print("Risk samples: ", len(Y[Y==1]))
        print("Safe samples: ", len(Y[Y==0]))

        
        pred, _ = risk_estimator.sample(X)
        print(f"Accuracy on train data: {100 * (Y == pred).mean()}%")
        
        X, Y = RiskEstimationDataset.dataloader_to_array(
            all_trials_test_dataloader)

        Y = Y.cpu().numpy().squeeze()
        pred, _ = risk_estimator.sample(X)
        print(f"Accuracy on all test data: {100 * (Y == pred).mean()}%")
    
    if plotter:
        pp_matrix_from_data(Y, pred, columns=["Safe", "Danger"])
        


def main_data_aggregation(skill_name = "peg_place"):
    n = number_of_saved_trials(skill_name)
    video_names = [f"{skill_name}_trial_{i}" for i in range(n)]

    data_aggregation(skill_name, video_names)


def test_data_aggregation(skill_name = "peg_door"):
    set_session("test_session")
    n = number_of_saved_trials(skill_name)
    video_names = [f"{skill_name}_trial_{i}" for i in range(n)]

    data_aggregation(skill_name, video_names[0:1])
    
    

if __name__ == '__main__':
    test_risk_flag_extremes()
    main_data_aggregation()

