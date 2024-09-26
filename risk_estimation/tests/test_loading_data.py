import numpy as np
import risk_estimation
from risk_estimation.models.risk_estimation.frame_dropping import (
    NoFrameDroppingPolicy,
    ProactiveRiskLabelingDroppingPolicy,
    OnlyLabelledFramesDroppingPolicy,
)
from risk_estimation.models.risk_estimation.risk_dataloader import (
    RiskEstimationDataset,
)

from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, VideoObservationsRiskLabels
from risk_estimation.models.risk_estimator import MLPRiskEstimator
import video_embedding
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_trial_names, behaviour_trial_names, set_session


def test_loading_of_risk_aware_data_from_videos(
    frame_dropping_policy=NoFrameDroppingPolicy,
):
    set_session("test_session")
    video_names = all_trial_names('peg_door')
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/", name='peg_door',
    )

    print("Manual dataset load")
    print("1. load dataloader")
    dataloaders = RiskEstimationDataset.load_videos_data_dataloaders(
        video_names, batch_size=10000, frame_dropping_policy=NoFrameDroppingPolicy
    )
    print(dataloaders)

    X = []
    Y = []
    print("2. dataloader to array + extract features")
    for dl in dataloaders:
        X_, Y_ = RiskEstimationDataset.dataloader_extract_to_array(
            dl, features=VideoObservationsRiskLabels
        )
        X.append(X_.cpu().numpy())
        Y.append(Y_.cpu().numpy())
        
    X = np.vstack(X)
    Y = np.vstack(Y)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print()

    print("Automatic dataset load")
    for frame_dropping_policy in [
        NoFrameDroppingPolicy,
        OnlyLabelledFramesDroppingPolicy,
        ProactiveRiskLabelingDroppingPolicy
        ]:
        for feature_extraction in [
            LatentObservationsRiskLabels,
            VideoObservationsRiskLabels,
            StampedLatentObservationsRiskLabels,
            StampedDistLatentObservationsRiskLabels,
        ]:
            print(f"Frame dropping policy: {frame_dropping_policy}")
            print(f"Features: {feature_extraction}")
            dataset = RiskEstimationDataset.load_dataset(
                video_names,
                video_embedder,
                frame_dropping_policy=frame_dropping_policy,
                features=feature_extraction,
            )
            print(f"X shape: {dataset.X.shape}, Y shape: {dataset.Y.shape}")
            print()


    print("Load dataset divided into train / test")
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names, video_embedder, video_embedder.batch_size, frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
    )
    X_train, Y_train = RiskEstimationDataset.dataloader_extract_to_array(train_dl)
    X_test, Y_test = RiskEstimationDataset.dataloader_extract_to_array(test_dl)
    print(f"train: X shape: {X_train.shape}, Y shape: {Y_train.shape}")
    print(f"test:  X shape: {X_test.shape}, Y shape: {Y_test.shape}")
    print(f"test:  Risky: {len(Y_train[Y_train==1])}, Safe {len(Y_train[Y_train==0])}")

def test_load_video_names_of_specific_behaviours():
    set_session("manipulation_demo_session")
    video_names = behaviour_trial_names('peg_pick', behaviours=['successful','peg_rotated','cables','hands'])
    
    assert video_names == ['peg_pick', 'peg_pick_trial_0', 'peg_pick_trial_1', 'peg_pick_trial_2', 'peg_pick_trial_3', 'peg_pick_trial_4']

    video_names = behaviour_trial_names('peg_pick', behaviours=['successful'])

    assert video_names == ['peg_pick', 'peg_pick_trial_0']

    video_names = behaviour_trial_names('peg_pick', behaviours=['applied_force'])

    assert video_names == ['peg_pick_trial_5']
    
    video_names = behaviour_trial_names('peg_pick', behaviours=[])

    assert video_names == []


if __name__ == "__main__":
    test_loading_of_risk_aware_data_from_videos()
