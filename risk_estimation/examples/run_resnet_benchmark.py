
from video_embedding.utils import set_session
from torch.utils.data import DataLoader
from risk_estimation.models.safety_layer import get_risk_estimator
from risk_estimation.datasets.frame_dropping import NoFrameDroppingPolicy
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset as D
from risk_estimation.datasets.risk_feature_extractor import *
from risk_estimation.datasets.frame_dropping import *
from risk_estimation.result_evaluator import benchmark_eval_save
from risk_estimation.scripts.result_img_save import sample_and_save_on_video, video_triplets_save
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_test_names, all_trial_names

def run_resnet_benchmarks(
        skill_name: str,
        embedding_approach: str,
        video_latent_dim: int, # size of latent dimension
        approach: str, # "TwinGP" or "MLP2 or "Resnet-50"
        features: str = "LatentObservationsRiskLabels", # StampedLatent (h+alpha) (is default)
        framedrop_policy: str = f"OnlyLabelledFramesDroppingPolicy", 
        train_epoch: int = 500, 
        train_patience: int = 500, 
    ):
    
    if isinstance(features, str):
        features = eval(features)

    if isinstance(framedrop_policy, str):
        framedrop_policy = eval(framedrop_policy)
    
    features=eval("Resnet"+features().__class__.__name__)

    video_embedder = VideoEmbedder(
        name=skill_name,
        latent_dim=video_latent_dim,
        nn_model=embedding_approach,
    )

    
    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, None, train_patience, train_epoch
    )

    train_dataloader = D.load_dataloader(all_trial_names(skill_name), video_embedder, 64, framedrop_policy, features, add_whiteblackimg=False)#, transform=D.resnet_transform)
    test_dataloader = D.load_dataloader(all_test_names(skill_name), video_embedder, 64, framedrop_policy, features, add_whiteblackimg=False)#, transform=D.resnet_transform)
    # novel_dataloader = D.load_dataloader(all_test_names(skill_name), video_embedder, 64, framedrop_policy.novel(), features.novel(), transform=D.resnet_transform)

    risk_estimator.set_dataloaders_for_validation([test_dataloader, 
        # novel_dataloader
        ], names=["test", 
        # "novel"
        ])
    
    risk_estimator.training_loop(train_dataloader, early_stop=True)
    # risk_estimator.save_model()
    # risk_estimator.load_model()


    benchmark_eval_save("Train_dataset", skill_name, train_dataloader.dataset, video_embedder, risk_estimator)
    benchmark_eval_save("Test_dataset", skill_name, test_dataloader.dataset, video_embedder, risk_estimator)
    # benchmark_eval_save("Novel_dataset", skill_name, novel_dataloader.dataset, video_embedder, risk_estimator)
    
    # for video_name in train_dataloader.dataset.video_names + test_dataloader.dataset.video_names:
    #     sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, 
    #                              train_dataloader, folder="autogen")
    #     video_triplets_save(video_name, video_embedder, folder="autogen")



skills =[
    # "peg_pick404", 
    "peg_door404", 
    # "slider_move404", 
    # "slider_move404_2", # has better alignment between train and test trajectory data
    # "peg_place404", 
    # "probe_pick404"
    # "move_around404"
]
session = "AE3"
stages = [
    # [64,    "CustomResnetStage1"],  # stage 1, # needs stage-1 latent dim
    [256,   "CustomResnetStage2"],  # stage 2, # needs stage-2 latent dim
    [512,   "CustomResnetStage3"],  # stage 3, # needs stage-3 latent dim
    [1024,  "CustomResnetStage4"], # stage 4, # needs stage-4 latent dim
    [2048,  "CustomResnetStage5"], # stage 5, # needs stage-5 latent dim
]
approach = "TwinGP"
approach = "MLP2"
# approach = "Resnet-50" # TODO: ResNet is the estimator and embedder

set_session(session)
for skill_name in skills:
    for video_latent_dim,embedding_approach in stages:
        run_resnet_benchmarks(
            skill_name = skill_name,
            embedding_approach = embedding_approach,
            video_latent_dim = video_latent_dim,
            approach = approach,
        )
