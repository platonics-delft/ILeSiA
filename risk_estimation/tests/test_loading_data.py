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
from video_embedding.utils import all_trial_names, all_test_names, set_session

from torch.utils.data import DataLoader
from risk_estimation.models.safety_layer import get_risk_estimator


def test_loading(
        skill_name='peg_door404',
        video_latent_dim=12,
        approach='TwinGP',
        embedding_approach="Autoencoder2",
        features="StampedLatentObservationsRiskLabels",
        framedrop_policy=f"OnlyLabelledFramesDroppingPolicy",
        out_assessment="cautious",
        train_epoch=1000,
        train_patience=2000,
    ):
    import time
    t = time.time()
    set_session("quantitative_study")
    print(f"TTT1 {time.time()-t}")
    t = time.time()

    video_names = all_trial_names('peg_door404')
    print(f"TTT2 {time.time()-t}")
    t = time.time()

    video_embedder = VideoEmbedder(name=skill_name, latent_dim=video_latent_dim)
    print(f"TTT3 {time.time()-t}")
    t = time.time()

    video_embedder.load_model()
    print(f"TTT4 {time.time()-t}")
    t = time.time()


    if isinstance(features, str):
        features = eval(features)

    print(f"TTT5 {time.time()-t}")
    t = time.time()

    if isinstance(framedrop_policy, str):
        framedrop_policy = eval(framedrop_policy)
    print(f"TTT6 {time.time()-t}")
    t = time.time()

    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )
    print(f"TTT7 {time.time()-t}")
    t = time.time()

    video_train_names = all_trial_names(skill_name)
    print(f"TTT8 {time.time()-t}")
    t = time.time()

    video_test_names = all_test_names(skill_name)

    print(f"TTT9 {time.time()-t}")
    t = time.time()


    dataset_nodrop = RiskEstimationDataset.load_dataset(video_train_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy, features=features)
    print(f"TTT10 {time.time()-t}")
    t = time.time()

    train_dataset, train_imgset, test_dataset, test_imgset = RiskEstimationDataset.extended_load(
        video_train_names, video_test_names, video_embedder, 
        framedrop_policy, features,
    )

    print(f"TTT11 {time.time()-t}")
    t = time.time()
    # optional, not aligned with other dataloaders
    risk_estimator.dataloader_test_for_plot = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)
    print(f"TTT12 {time.time()-t}")
    t = time.time()

    risk_estimator.dataloader_nodrop_for_plot = DataLoader(dataset_nodrop, batch_size=video_embedder.batch_size, shuffle=True)
    print(f"TTT13 {time.time()-t}")
    t = time.time()



if __name__ == "__main__":
    test_loading()
