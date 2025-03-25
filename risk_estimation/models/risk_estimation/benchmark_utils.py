#!/usr/bin/env python3
from typing import Iterable, List

import torchvision
from risk_estimation.models.safety_layer import SafetyLayer
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, VideoObservationsRiskLabels, LatentObservationsRiskLabelsPriorRisk, StampedDistLatentObservationsRiskLabelsPriorRisk, StampedLatentObservationsRiskLabelsPriorRisk
import video_embedding, risk_estimation
from torch.utils.data import DataLoader
from video_embedding.models.video_embedder import VideoEmbedder, VideoEmbedder

from video_embedding.utils import all_test_names, all_trial_names, get_session, set_session

from risk_estimation.models.risk_estimator import sample_and_save_on_video
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy

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

