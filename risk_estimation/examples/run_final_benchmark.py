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
        approach: str,
        embedding_approach: str,
        video_latent_dim: int = 12, # size of latent dimension
        features: str = "StampedLatentObservationsRiskLabels", # StampedLatent (h+alpha) (is default)
        framedrop_policy: str = f"OnlyLabelledFramesDroppingPolicy", 
        out_assessment: str = "cautious", # "cautious" = (mu+sigma) (is default) or "optimistic" = (mu)
        train_epoch: int = 500, 
        train_patience: int = 500, 
        save_video_flag: bool = True, # if True, save the video triplets
        add_whiteblackimg: bool = True, # if True, add white and black images to the dataset as risky
        filter_samples_to_label_areas: bool = False, # if True, only use samples in the label areas
    ):
    if filter_samples_to_label_areas: framedrop_policy += f"Risk{skill_name}"
    framedrop_policy = eval(framedrop_policy)
    features = eval(features)
    
    video_embedder = VideoEmbedder(name=skill_name, latent_dim=video_latent_dim, nn_model=embedding_approach)
    video_embedder.load_model()
    
    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )

    train_dataloader = D.load_dataloader(all_trial_names(skill_name), video_embedder, 64, framedrop_policy, features, add_whiteblackimg=add_whiteblackimg)
    novel_dataloader = D.load_dataloader(all_test_names(skill_name), video_embedder, 64, framedrop_policy.novel(), features.novel(), add_whiteblackimg=add_whiteblackimg)
    test_dataloader = D.load_dataloader(all_test_names(skill_name), video_embedder, 64, framedrop_policy, features, add_whiteblackimg=add_whiteblackimg)
    
    # optional, view accuracy on these validation dataloaders
    risk_estimator.set_dataloaders_for_validation([
        test_dataloader, 
        novel_dataloader
    ], names=["test", "novel"])

    risk_estimator.training_loop(train_dataloader, early_stop=True)
    risk_estimator.save_model()
    # risk_estimator.load_model()

    benchmark_eval_save("Train_dataset", skill_name, train_dataloader.dataset, video_embedder, risk_estimator)
    benchmark_eval_save("Test_dataset", skill_name, test_dataloader.dataset, video_embedder, risk_estimator)
    benchmark_eval_save("Novel_Dataset", skill_name, novel_dataloader.dataset, video_embedder, risk_estimator)
    
    for video_name in train_dataloader.dataset.video_names + test_dataloader.dataset.video_names:
        sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, train_dataloader, folder="autogen")
        if save_video_flag:
            video_triplets_save(video_name, video_embedder, folder="autogen")

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
    # 'MLP2',
    # 'GP',
    'TwinGP',
]
set_session("AE3")

for skill_name in skills:
    for approach in approaches:
        run_final_benchmarks(
            skill_name=skill_name,
            approach=approach,
            embedding_approach="Autoencoder3",
            train_epoch=500,
        )


