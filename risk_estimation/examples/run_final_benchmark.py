

from torch.utils.data import DataLoader
from risk_estimation.models.safety_layer import get_risk_estimator
from risk_estimation.datasets.frame_dropping import NoFrameDroppingPolicy
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset as D
from risk_estimation.datasets.risk_feature_extractor import *
from risk_estimation.datasets.frame_dropping import *
from risk_estimation.result_evaluator import benchmark_eval_save
from risk_estimation.scripts.result_img_save import sample_and_save_on_video, video_triplets_save
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_test_names, all_trial_names, set_session

def run_final_benchmarks(
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
        add_whiteblackimg: bool,
    ):
    if isinstance(features, str): features = eval(features)
    if isinstance(framedrop_policy, str): framedrop_policy = eval(framedrop_policy)
    
    video_embedder = VideoEmbedder(name=skill_name, latent_dim=video_latent_dim, nn_model=embedding_approach)
    video_embedder.load_model()
    
    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )

    train_dataloader = D.load_dataloader(all_trial_names(skill_name), video_embedder, video_embedder.batch_size, framedrop_policy, features, add_whiteblackimg=add_whiteblackimg)
    test_dataloader = D.load_dataloader(all_test_names(skill_name), video_embedder, video_embedder.batch_size, framedrop_policy, features, add_whiteblackimg=add_whiteblackimg)
    
    # optional, view accuracy on these validation dataloaders
    risk_estimator.set_dataloaders_for_validation([
        test_dataloader, 
        D.load_dataloader(all_trial_names(skill_name), video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy, features=features)
    ], names=["test", "nodrop"])

    risk_estimator.training_loop(train_dataloader, early_stop=True)
    risk_estimator.save_model()
    # risk_estimator.load_model()

    benchmark_eval_save("Train_dataset", skill_name, train_dataloader.dataset, video_embedder, risk_estimator)
    benchmark_eval_save("Test_dataset", skill_name, test_dataloader.dataset, video_embedder, risk_estimator)

    # Additional no drop eval
    # nds_train_dataset, nds_test_dataset = D.extended_load(train_dataset.video_names, test_dataset.video_names, video_embedder, NoFrameDroppingPolicy, features, resnet_type_risk_estimator)
    # benchmark_eval_save("NoDrop_Prior_is_Safe", skill_name, nds_train_dataset, video_embedder, risk_estimator)
    
    for video_name in train_dataloader.dataset.video_names + test_dataloader.dataset.video_names:
        sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, train_dataloader, folder="autogen")
        if save_video_flag:
            video_triplets_save(video_name, video_embedder, folder="autogen")

mapping = {
    "peg_pick404": "PegPick", 
    "peg_door404": "PegDoor", 
    "slider_move404": "SliderMove", 
    "slider_move404_2": "SliderMove", # has better alignment between train and test trajectory data
    "peg_place404": "PegPlace", 
    "probe_pick404": "ProbePick",
    "move_around404": "MoveAround",
}

skills = [
    "peg_pick404", 
    # "peg_door404", 
    # "peg_place404", 
    # "slider_move404", 
    # "slider_move404_2", # has better alignment between train and test trajectory data
    # "probe_pick404"
    # "move_around404"
] 

approaches = [
    # 'LR',
    # 'MLP',
    'MLP2',
    # 'GP',
    # 'TwinGP',
]
filter_samples_to_label_areas = False
add_whiteblackimg = True



for skill_name in skills:
    for approach in approaches:
        # set_session("quantitative_study")
        set_session("AE3")
        if filter_samples_to_label_areas:
            framedrop_policy = f"OnlyLabelledFramesDroppingPolicyRisk{mapping[skill_name]}"
        else:
            framedrop_policy = f"OnlyLabelledFramesDroppingPolicy"
        # framedrop_policy = f"NoFrameDroppingPolicy"
        run_final_benchmarks(
            skill_name=skill_name,
            video_latent_dim=12,
            approach=approach,
            embedding_approach="Autoencoder2",
            features="StampedLatentObservationsRiskLabels",
            framedrop_policy=framedrop_policy,
            out_assessment="cautious",
            train_epoch=500,
            train_patience=6000,
            save_video_flag=True,
            add_whiteblackimg=add_whiteblackimg,
        )


