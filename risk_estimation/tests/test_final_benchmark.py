#!/usr/bin/env python3
import argparse
from typing import Iterable, List

import torchvision
from risk_estimation.models.safety_layer import SafetyLayer, get_risk_estimator
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import (
    LatentObservationsRiskLabels, 
    StampedDistLatentObservationsRiskLabels, 
    StampedLatentObservationsRiskLabels, 
    VideoObservationsRiskAndSafeLabels, 
    VideoObservationsRiskLabels, 
    LatentObservationsRiskLabelsPriorRisk, 
    StampedDistLatentObservationsRiskLabelsPriorRisk, 
    StampedLatentObservationsRiskLabelsPriorRisk, 
    ResnetLatentObservationsRiskLabels,
    StampedDistRecErrLatentObservationsRiskLabels
)
from risk_estimation.models.risk_estimation.benchmark_utils import (
    get_dataset_and_images_nodrop, 
    benchmark_eval_save,
    check_validity
)
import video_embedding, risk_estimation
from torch.utils.data import DataLoader
from video_embedding.models.video_embedder import VideoEmbedder, RiskyBehavioralVideoEmbedder

from video_embedding.utils import all_test_names, all_trial_names, get_session, set_session
from risk_estimation.models.risk_estimator import sample_and_save_on_video, video_triplets_save
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
from risk_estimation.models.risk_estimation.frame_dropping import (
    NoFrameDroppingPolicy, 
    OnlyLabelledFramesDroppingPolicy, 
    OnlyLabelledFramesDroppingPolicyRiskPegPick1,
    OnlyLabelledFramesDroppingPolicyRiskPegPick2,
    OnlyLabelledFramesDroppingPolicyRiskPegDoor1,
    OnlyLabelledFramesDroppingPolicyRiskPegDoor2,
    OnlyLabelledFramesDroppingPolicyRiskPegPlace1,
    OnlyLabelledFramesDroppingPolicyRiskPegPlace2,
    OnlyLabelledFramesDroppingPolicyRiskSliderMove1,
    OnlyLabelledFramesDroppingPolicyRiskSliderMove2,
    OnlyLabelledFramesDroppingPolicyRiskMoveAround1,
    OnlyLabelledFramesDroppingPolicyRiskMoveAround2,

)
import pandas as pd
import torch

import time

mapping = {
    "peg_pick404": "PegPick", 
    "peg_door404": "PegDoor", 
    "slider_move404": "SliderMove", 
    "slider_move404_2": "SliderMove", # has better alignment between train and test trajectory data
    "peg_place404": "PegPlace", 
    "probe_pick404": "ProbePick",
    "move_around404": "MoveAround",
}

class TimeIt():
    def __init__(self):
        self.t_prev = 0.0
    def __call__(self, name):
        print(f"Time: {name} {time.perf_counter()-self.t_prev}")
        self.t_prev = time.perf_counter()

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

    video_embedder = RiskyBehavioralVideoEmbedder(
        name=skill_name,
        latent_dim=video_latent_dim,
        nn_model=embedding_approach,
    )

    if not resnet_type_embedding_approach: # embedding approach model is not type resnet
        video_embedder.load_model()
    
    # DEMO DAY: Because of very long skills recorded > 700 frames, employing two separate risk estimation models
    risk_estimator = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )
    risk_estimator2 = get_risk_estimator(
        approach, skill_name, features.xdim(video_latent_dim), video_embedder, out_assessment, train_patience, train_epoch
    )


    video_train_names = all_trial_names(skill_name)
    video_test_names = all_test_names(skill_name)

    dataset_nodrop = RiskEstimationDataset.load_dataset(video_train_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy, features=features)

    risk = "1"
    framedrop_policy = eval(f"{OnlyLabelledFramesDroppingPolicy().__class__.__name__}Risk{mapping[skill_name]}{risk}")
    train_dataset, train_imgset, test_dataset, test_imgset = RiskEstimationDataset.extended_load(
        video_train_names, video_test_names, video_embedder, 
        framedrop_policy, features, resnet_option=resnet_type_risk_estimator
    )

    risk_estimator.dataloader_test_for_plot = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)
    risk_estimator.dataloader_nodrop_for_plot = DataLoader(dataset_nodrop, batch_size=video_embedder.batch_size, shuffle=True)
    risk_estimator.training_loop(DataLoader(train_dataset, batch_size=video_embedder.batch_size), early_stop=True)
    risk_estimator.save_model()


    risk = "2"
    framedrop_policy = eval(f"{OnlyLabelledFramesDroppingPolicy().__class__.__name__}Risk{mapping[skill_name]}{risk}")
    train_dataset, train_imgset, test_dataset, test_imgset = RiskEstimationDataset.extended_load(
        video_train_names, video_test_names, video_embedder, 
        framedrop_policy, features, resnet_option=resnet_type_risk_estimator
    )
    risk_estimator2.dataloader_test_for_plot = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)
    risk_estimator2.dataloader_nodrop_for_plot = DataLoader(dataset_nodrop, batch_size=video_embedder.batch_size, shuffle=True)
    risk_estimator2.training_loop(DataLoader(train_dataset, batch_size=video_embedder.batch_size), early_stop=True)
    risk_estimator2.save_model()


    benchmark_eval_save("Train_dataset", skill_name, train_dataset, train_imgset, video_embedder, risk_estimator)
    benchmark_eval_save("Test_dataset", skill_name, test_dataset, test_imgset, video_embedder, risk_estimator)
    
    
    # Additional no drop eval
    # nds_train_dataset, nds_train_imgset, nds_test_dataset, nds_test_imgset = RiskEstimationDataset.extended_load(video_train_names, video_test_names, video_embedder, NoFrameDroppingPolicy, features, resnet_type_risk_estimator)
    # ndr_train_dataset, ndr_train_imgset, ndr_test_dataset, ndr_test_imgset = RiskEstimationDataset.extended_load(video_train_names, video_test_names, video_embedder, NoFrameDroppingPolicy, eval(features().__class__.__name__+"PriorRisk"), resnet_type_risk_estimator)

    # benchmark_eval_save("NoDrop_Prior_is_Safe", skill_name, nds_train_dataset, nds_train_imgset, video_embedder, risk_estimator)
    # benchmark_eval_save("NoDrop_Prior_is_Risk", skill_name, ndr_train_dataset, ndr_train_imgset, video_embedder, risk_estimator)

    
    for video_name in video_train_names:
        sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, 
                                 DataLoader(train_dataset, batch_size=video_embedder.batch_size), folder="autogen")
        if save_video_flag and not resnet_type_embedding_approach:
            video_triplets_save(video_name, video_embedder, risk_estimator, features, 
                                 DataLoader(train_dataset, batch_size=video_embedder.batch_size), folder="autogen")
    for video_name in video_test_names:
        sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, 
                                 DataLoader(train_dataset, batch_size=video_embedder.batch_size), folder="autogen")
        if save_video_flag and not resnet_type_embedding_approach:
            video_triplets_save(video_name, video_embedder, risk_estimator, features, 
                                 DataLoader(train_dataset, batch_size=video_embedder.batch_size), folder="autogen")


def test_all_final_benchmarks(args):
    if isinstance(args.framedrop, str):
        framedrop = eval(args.framedrop)
    else:
        framedrop = args.framedrop

    for session in args.session:
        set_session(session)
        for skill_name in args.skill:
            save_video_flag = True
            for out_assessment in args.out_assessment:
                for approach in args.approach:
                    for video_latent_dim in args.latent_dim:
                        for features in args.features:

                            if isinstance(features, str):
                                features = eval(features)

                            if not check_validity(approach, features): continue
                            print("save_video_flag ", save_video_flag)
                            test_final_benchmarks(
                                skill_name = skill_name,
                                video_latent_dim = video_latent_dim,
                                approach = approach,
                                embedding_approach = args.embedding_approach,
                                features = features,
                                framedrop_policy = framedrop,
                                out_assessment = out_assessment,
                                train_epoch = args.epoch,
                                train_patience = args.patience,
                                save_video_flag = save_video_flag,
                            )
                            save_video_flag = False
                            print("=== test finished ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Final benchmark",
        description="",
        epilog="",
    )

    parser.add_argument("--skill", nargs="+", default=[
        # "peg_pick404", 
        "peg_door404", 
        # "slider_move404", 
        # "slider_move404_2", # has better alignment between train and test trajectory data
        # "peg_place404", 
        # "probe_pick404"
        # "move_around404"
    ])
    parser.add_argument("--session", nargs="+", default=[
        # "manipulation_demo404_session", # no augmentation
        # "manipulation_demo404_conbr_session", # contrast brightness augmentation, video embed., trained with 1200 epochs 
        # "manipulation_demo404_augment_session",  # contrast brightness, image move 1pix, perspective scale 0.1
        "manipulation_demo404_augment_12_session", # contrast 0.05 brightness 0.05, image move 0.7pix, perspective scale 0.04
        # "manipulation_demo404_augment_24_session", # contrast 0.05 brightness 0.05, image move 0.7pix, perspective scale 0.04
        # "manipulation_demo404_augment_32_session", # contrast 0.05 brightness 0.05, image move 0.7pix, perspective scale 0.04
        # "manipulation_demo404_augment_48_session", # contrast 0.05 brightness 0.05, image move 0.7pix, perspective scale 0.04
        # "manipulation_demo404_augment_64_session", # contrast 0.05 brightness 0.05, image move 0.7pix, perspective scale 0.04
    ])
    parser.add_argument("--latent_dim", nargs="+", default=[
        # 8,
        12,
        # 16,
        # 24,
        # 32,
        # 48,
        # 64,
        # 64,  # stage 1
        # 256,  # stage 2
        # 512,  # stage 3
        # 1024, # stage 4
        # 2048, # stage 5
    ])
    parser.add_argument("--learning_rate", default=0.01)
    parser.add_argument(
        "--framedrop", 
        default=OnlyLabelledFramesDroppingPolicy,
        # default=OnlyLabelledFramesDroppingPolicyRiskPegPick1,
        # default=OnlyLabelledFramesDroppingPolicyRiskPegPick2,
        # default=OnlyLabelledFramesDroppingPolicyRiskPegDoor1,
        # default=OnlyLabelledFramesDroppingPolicyRiskPegDoor2,
        # default=OnlyLabelledFramesDroppingPolicyRiskPegPlace1,
        # default=OnlyLabelledFramesDroppingPolicyRiskPegPlace2,
        # default=OnlyLabelledFramesDroppingPolicyRiskSliderMove1,
        # default=OnlyLabelledFramesDroppingPolicyRiskSliderMove2,
        # default=OnlyLabelledFramesDroppingPolicyRiskMoveAround1,
        # default=OnlyLabelledFramesDroppingPolicyRiskMoveAround2,
    )
    parser.add_argument(
        "--epoch", 
        default=6000,
        type=int,
    )
    parser.add_argument(
        "--patience", 
        default=6000,
        type=int,
    )
    parser.add_argument(
        "--approach", nargs="+", # approaches
        default = [
            # 'LR',
            # 'MLP',
            'GP',
            # "L+GP",
            # "L+GP+1SKIP",
            # "L+GP+2SKIP",
            # 'resnet50',
        ],
    )
    parser.add_argument(
        "--embedding_approach", 
        default = "Autoencoder",
        # default = "LargeAutoencoder",
        # default = "CustomResnetStage1", # needs stage-1 latent dim
        # default = "CustomResnetStage2", # needs stage-2 latent dim
        # default = "CustomResnetStage3", # needs stage-3 latent dim
        # default = "CustomResnetStage4", # needs stage-4 latent dim
        # default = "CustomResnetStage5", # needs stage-5 latent dim
    )
    parser.add_argument(
        "--out_assessment", nargs="+",
        default = [
            # 'optimistic',
            'cautious',
        ],
    )
    parser.add_argument(
        "--features", nargs="+",
        default = [
            # LatentObservationsRiskLabels,
            # StampedLatentObservationsRiskLabels,
            StampedDistLatentObservationsRiskLabels,
            # StampedDistRecErrLatentObservationsRiskLabels,
        ],
    )


    test_all_final_benchmarks(parser.parse_args())


