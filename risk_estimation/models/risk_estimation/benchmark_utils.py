#!/usr/bin/env python3
from typing import Iterable, List

import torchvision
from risk_estimation.models.safety_layer import SafetyLayer
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, VideoObservationsRiskLabels, LatentObservationsRiskLabelsPriorRisk, StampedDistLatentObservationsRiskLabelsPriorRisk, StampedLatentObservationsRiskLabelsPriorRisk
import video_embedding, risk_estimation
from torch.utils.data import DataLoader
from video_embedding.models.video_embedder import VideoEmbedder, RiskyBehavioralVideoEmbedder

from video_embedding.utils import all_test_names, all_trial_names, get_session, set_session

from risk_estimation.models.risk_estimator import sample_and_save_on_video
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
import pandas as pd


def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_results_to_csv(results):
    """ DEPRECATED """
    flatten_results = flatten_dict(results)

    data_list = []
    for key, results in flatten_results.items():
        
        out_assessment_, approach_, latent_dim_, features_, skill_ = key.split('.')
        data_list.append({
            'Skill': skill_,
            'Assessment': out_assessment_,
            'Approach': approach_,
            'L.Dim': latent_dim_,
            'Features': features_,
            'Test': results[0],
            'Train': results[1],
            'NolabelPriorSafe': results[2],
            'NolabelPriorDangerous': results[3],
        })

    df = pd.DataFrame(data_list)

    df.to_markdown(f"{risk_estimation.path}/results.md")

    print(results)


def get_dataset_and_images_nodrop(video_embedder):
    """ DEPRECATED """
    dataset_nodrop = RiskEstimationDataset.load_dataset(video_train_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=features,)
    dataloader_nodrop = DataLoader(dataset_nodrop, batch_size=40, shuffle=False)

    risk_estimator.dataloader_test_for_plot = test_type1_dataloader
    risk_estimator.dataloader_nodrop_for_plot = dataloader_nodrop




def benchmark_eval_save(
        title,
        skill_name,
        dataset,
        imgset,
        video_embedder,
        risk_estimator,
    ):
    path = f"{risk_estimation.path}/autogen/{get_session()}/{skill_name}/"

    e = ResultEvaluator(name=f"{title}_{risk_estimator.encode_params_as_str()}", savepath=path, iwanttosee=["accuracy"])
    e(risk_estimator, video_embedder, dataset.X, dataset.Y, imgset.X, imgset.Y)


def check_validity(features, approach):
    if approach in ['DistLS', 'DistLR', 'DistMin']:
        if features.xreq != ['image', 'frame_number']:
            return False
    if approach == "L+GP+1SKIP":
        if features.xreq != ['image', 'frame_number'] and features.xreq != ['image', 'frame_number', 'similarity_dist']:
            return False
    if approach == "L+GP+2SKIP":
        if features.xreq != ['image', 'frame_number', 'similarity_dist']:
            return False

    return True
