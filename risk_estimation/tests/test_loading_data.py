from risk_estimation.datasets.frame_dropping import *
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset
from risk_estimation.datasets.risk_feature_extractor import *
from risk_estimation.models.safety_layer import get_risk_estimator
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_trial_names, all_test_names, set_session

from torch.utils.data import DataLoader

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
    set_session("quantitative_study")

    video_names = all_trial_names('peg_door404')

    video_embedder = VideoEmbedder(name=skill_name, latent_dim=video_latent_dim)

    video_embedder.load_model()


    if isinstance(features, str):
        features = eval(features)

    if isinstance(framedrop_policy, str):
        framedrop_policy = eval(framedrop_policy)

    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )

    video_train_names = all_trial_names(skill_name)

    video_test_names = all_test_names(skill_name)



    dataset_nodrop = RiskEstimationDataset.load_dataset(video_train_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy, features=features)

    train_dataset, train_imgset, test_dataset, test_imgset = RiskEstimationDataset.extended_load(
        video_train_names, video_test_names, video_embedder, 
        framedrop_policy, features,
    )

    # optional, not aligned with other dataloaders
    risk_estimator.dataloader_test_for_plot = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)

    risk_estimator.dataloader_nodrop_for_plot = DataLoader(dataset_nodrop, batch_size=video_embedder.batch_size, shuffle=True)



if __name__ == "__main__":
    test_loading()
