

from torch.utils.data import DataLoader
from risk_estimation.models.safety_layer import get_risk_estimator
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import *
from risk_estimation.models.risk_estimation.result_evaluator import benchmark_eval_save
from risk_estimation.models.risk_estimation.frame_dropping import *
from risk_estimation.models.risk_estimator import sample_and_save_on_video, video_triplets_save
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_test_names, all_trial_names

def test_final_benchmarks(
        skill_name: str,
        video_latent_dim: int,
        approach: str,
        embedding_approach: str,
        features: str,
        framedrop_policy,
        out_assessment: str,
        train_epoch: int,
        train_patience: int,
        save_video_flag: bool,
    ):
    
    if isinstance(features, str):
        features = eval(features)

    if isinstance(framedrop_policy, str):
        framedrop_policy = eval(framedrop_policy)
    
    resnet_type_risk_estimator = ('resnet' in approach)
    resnet_type_embedding_approach = ("Resnet" in embedding_approach)
    if resnet_type_risk_estimator:
        features=eval("Resnet"+features().__class__.__name__)

    video_embedder = VideoEmbedder(
        name=skill_name,
        latent_dim=video_latent_dim,
        nn_model=embedding_approach,
    )

    if not resnet_type_embedding_approach: # embedding approach model is not type resnet
        video_embedder.load_model()
    
    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )

    video_train_names = all_trial_names(skill_name)
    video_test_names = all_test_names(skill_name)

    print("video_train_names: ", video_train_names, " video_test_names: ", video_test_names)

    dataset_nodrop = RiskEstimationDataset.load_dataset(video_train_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy, features=features)

    train_dataset, train_imgset, test_dataset, test_imgset = RiskEstimationDataset.extended_load(
        video_train_names, video_test_names, video_embedder, 
        framedrop_policy, features, resnet_option=resnet_type_risk_estimator
    )

    # optional, not aligned with other dataloaders
    risk_estimator.dataloader_test_for_plot = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)
    risk_estimator.dataloader_nodrop_for_plot = DataLoader(dataset_nodrop, batch_size=video_embedder.batch_size, shuffle=True)

    risk_estimator.training_loop(DataLoader(train_dataset, batch_size=video_embedder.batch_size), early_stop=True)
    risk_estimator.save_model()
    # risk_estimator.load_model()


    benchmark_eval_save("Train_dataset", skill_name, train_dataset, train_imgset, video_embedder, risk_estimator)
    benchmark_eval_save("Test_dataset", skill_name, test_dataset, test_imgset, video_embedder, risk_estimator)

    
    # Additional no drop eval
    # nds_train_dataset, nds_train_imgset, nds_test_dataset, nds_test_imgset = RiskEstimationDataset.extended_load(video_train_names, video_test_names, video_embedder, NoFrameDroppingPolicy, features, resnet_type_risk_estimator)
    # ndr_train_dataset, ndr_train_imgset, ndr_test_dataset, ndr_test_imgset = RiskEstimationDataset.extended_load(video_train_names, video_test_names, video_embedder, NoFrameDroppingPolicy, eval(features().__class__.__name__+"PriorRisk"), resnet_type_risk_estimator)

    # benchmark_eval_save("NoDrop_Prior_is_Safe", skill_name, nds_train_dataset, nds_train_imgset, video_embedder, risk_estimator)
    # benchmark_eval_save("NoDrop_Prior_is_Risk", skill_name, ndr_train_dataset, ndr_train_imgset, video_embedder, risk_estimator)

    
    for video_name in video_train_names + video_test_names:
        sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, 
                                 DataLoader(train_dataset, batch_size=video_embedder.batch_size), folder="autogen")
        if save_video_flag and not resnet_type_embedding_approach:
            video_triplets_save(video_name, video_embedder, folder="autogen")

