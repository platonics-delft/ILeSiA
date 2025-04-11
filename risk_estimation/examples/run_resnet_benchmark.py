
from video_embedding.utils import set_session
from torch.utils.data import DataLoader
from risk_estimation.models.safety_layer import get_risk_estimator
from risk_estimation.datasets.frame_dropping import NoFrameDroppingPolicy
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset
from risk_estimation.datasets.risk_feature_extractor import *
from risk_estimation.datasets.frame_dropping import *
from risk_estimation.result_evaluator import benchmark_eval_save
from risk_estimation.scripts.result_img_save import sample_and_save_on_video, video_triplets_save
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_test_names, all_trial_names

def run_resnet_benchmarks(
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

    dataset_nodrop = RiskEstimationDataset.load_dataset(all_trial_names(skill_name), video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy, features=features)


    train_dataset, test_dataset = RiskEstimationDataset.extended_load(
        all_trial_names(skill_name), all_test_names(skill_name), video_embedder, 
        framedrop_policy, features, resnet_option=resnet_type_risk_estimator
    )
    print("video_train_names: ", train_dataset.video_names, " video_test_names: ", test_dataset.video_names)

    # optional, not aligned with other dataloaders
    risk_estimator.set_dataloaders_for_validation(test_dataset, dataset_nodrop)

    risk_estimator.training_loop(DataLoader(train_dataset, batch_size=video_embedder.batch_size), early_stop=True)
    risk_estimator.save_model()
    # risk_estimator.load_model()


    benchmark_eval_save("Train_dataset", skill_name, train_dataset, video_embedder, risk_estimator)
    benchmark_eval_save("Test_dataset", skill_name, test_dataset, video_embedder, risk_estimator)

    # Additional no drop eval
    # nds_train_dataset, nds_test_dataset = RiskEstimationDataset.extended_load(train_dataset.video_names, test_dataset.video_names, video_embedder, NoFrameDroppingPolicy, features, resnet_type_risk_estimator)
    # ndr_train_dataset, ndr_test_dataset = RiskEstimationDataset.extended_load(train_dataset.video_names, test_dataset.video_names, video_embedder, NoFrameDroppingPolicy, eval(features().__class__.__name__+"PriorRisk"), resnet_type_risk_estimator)

    # benchmark_eval_save("NoDrop_Prior_is_Safe", skill_name, nds_train_dataset, video_embedder, risk_estimator)
    # benchmark_eval_save("NoDrop_Prior_is_Risk", skill_name, ndr_train_dataset, video_embedder, risk_estimator)

    
    for video_name in train_dataset.video_names + test_dataset.video_names:
        sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, 
                                 DataLoader(train_dataset, batch_size=video_embedder.batch_size), folder="autogen")
        if save_video_flag and not resnet_type_embedding_approach:
            video_triplets_save(video_name, video_embedder, folder="autogen")



skills =[
    # "peg_pick404", 
    "peg_door404", 
    # "slider_move404", 
    # "slider_move404_2", # has better alignment between train and test trajectory data
    # "peg_place404", 
    # "probe_pick404"
    # "move_around404"
]
session = "quantitative_study"
stages = [
    [64,    "CustomResnetStage1"],  # stage 1, # needs stage-1 latent dim
    [256,   "CustomResnetStage2"],  # stage 2, # needs stage-2 latent dim
    [512,   "CustomResnetStage3"],  # stage 3, # needs stage-3 latent dim
    [1024,  "CustomResnetStage4"], # stage 4, # needs stage-4 latent dim
    [2048,  "CustomResnetStage5"], # stage 5, # needs stage-5 latent dim
]

set_session(session)
for skill_name in skills:
    save_video_flag = True
    for video_latent_dim,embedding_approach in stages:
        if isinstance(features, str):
            features = eval(features)

        print("save_video_flag ", save_video_flag)
        run_resnet_benchmarks(
            skill_name = skill_name,
            video_latent_dim = video_latent_dim,
            approach = 'resnet50',
            embedding_approach = embedding_approach,
            features = features,
            framedrop_policy = framedrop,
            out_assessment = out_assessment,
            train_epoch = 6000,
            train_patience = 6000,
            save_video_flag = save_video_flag,
        )
        save_video_flag = False