
#!/usr/bin/env python3
from typing import Iterable, List
from risk_estimation.models.safety_layer import SafetyLayer
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, LatentObservationsRiskLabelsPriorRisk, StampedDistLatentObservationsRiskLabelsPriorRisk, StampedLatentObservationsRiskLabelsPriorRisk
from risk_estimation.models.risk_estimator import (
    DistanceRiskEstimator,
    NMDistanceRiskEstimatorDTW,
    MLPRiskEstimator,
)
import video_embedding, risk_estimation
from torch.utils.data import DataLoader
from video_embedding.models.video_embedder import VideoEmbedder, RiskyBehavioralVideoEmbedder

from scipy.spatial.distance import cosine, euclidean
from video_embedding.utils import all_trial_names, get_session, set_session

from risk_estimation.models.risk_estimator import (
    DistanceRiskEstimator,
    LinSearchDistanceRiskEstimator,
    NMDistanceRiskEstimatorDTW,
    GPRiskEstimator,
    LRHyperTrainDistanceRiskEstimator,
    LRRiskEstimator,
    MLPRiskEstimator,
    MinHyperTrainDistanceRiskEstimator,
    sample_and_save_on_video,
)
from risk_estimation.models.risk_estimation.result_evaluator import ResultEvaluator
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
import pandas as pd


def test_all_final_benchmarks(
        final_benchmark_skills = ["peg_door"],
        session: str = "manipulation_demo_session",
        video_latent_dims: Iterable[int] = [4,8,16,32,64],
        approach: str = 'GP',
        features = LatentObservationsRiskLabels,
        framedrop_policy = OnlyLabelledFramesDroppingPolicy,
        out_assessment: str = 'optimistic',
    ):
    
    for video_latent_dim in video_latent_dims:
        res = test_final_benchmarks(
            final_benchmark_skills = final_benchmark_skills,
            session = session,
            video_latent_dim = video_latent_dim,
            approach = approach,
            features = features,
            framedrop_policy = framedrop_policy,
            out_assessment = out_assessment,
        )
    
        print(f"dim={video_latent_dim}: {res}")
    
def test_final_benchmarks(
                        final_benchmark_skills: List[str],
                        session: str,
                        video_latent_dim: int,
                        approach: str,
                        features,
                        framedrop_policy,
                        out_assessment: str,
                        ):
    

    set_session(session)
    results = {}
    for skill_name in final_benchmark_skills:
        video_embedder = RiskyBehavioralVideoEmbedder(name=skill_name, latent_dim=video_latent_dim)
        video_embedder.load_model()

        risk_estimator = GPRiskEstimator(name=skill_name, xdim=features.xdim(video_latent_dim), learning_rate=0.01, 
            out_assessment=out_assessment, train_patience=2000)

        video_names = all_trial_names(skill_name)

        train_dataloader, test_dataloader = RiskEstimationDataset.load(
            video_names=video_names,
            video_embedder=video_embedder,
            batch_size=video_embedder.batch_size,
            frame_dropping_policy=framedrop_policy,
            features=features,
        )
        train_dataloader_images, test_dataloader_images = RiskEstimationDataset.load(
            video_names=video_names,
            video_embedder=video_embedder,
            batch_size=video_embedder.batch_size,
            frame_dropping_policy=framedrop_policy,
            features=VideoObservationsRiskAndSafeLabels,
        )
        X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(test_dataloader_images)
        X_train_images, Y_train_images = RiskEstimationDataset.dataloader_to_array(train_dataloader_images)


        dataset_nodrop = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=features,)
        dataloader_nodrop = DataLoader(dataset_nodrop, batch_size=40, shuffle=False)

        print(f"Train with trials: {video_names}")
        
        risk_estimator.dataloader_test_for_plot = test_dataloader
        risk_estimator.dataloader_nodrop_for_plot = dataloader_nodrop
        risk_estimator.training_loop(train_dataloader, num_epochs=2000)

        path = f"{risk_estimation.path}/autogen/{get_session()}/{skill_name}/"

        X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dataloader)
        e = ResultEvaluator(name=f"Test dataset {risk_estimator.encode_params_as_str()}", savepath=path, iwanttosee=["accuracy"])
        wrong_test = e(risk_estimator, video_embedder, X_test, Y_test, X_test_images, Y_test_images)

        X_train, Y_train = RiskEstimationDataset.dataloader_to_array(train_dataloader)
        e = ResultEvaluator(name=f"Train dataset {risk_estimator.encode_params_as_str()}", savepath=path, iwanttosee=["accuracy"])
        wrong_train = e(risk_estimator, video_embedder, X_train, Y_train, X_train_images, Y_train_images)

        dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=features)
        dataset_images = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=VideoObservationsRiskAndSafeLabels)
        e = ResultEvaluator(name=f"NoDrop Prior is Safe {risk_estimator.encode_params_as_str()}", savepath=path, iwanttosee=["accuracy"])
        wrong_nodrop = e(risk_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)

        dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=eval(features().__class__.__name__+"PriorRisk"),)
        dataset_images = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=VideoObservationsRiskAndSafeLabels,)
        e = ResultEvaluator(name=f"NoDrop Prior is Risk {risk_estimator.encode_params_as_str()}", savepath=path, iwanttosee=["accuracy"])
        wrong_nodrop2 = e(risk_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)


        # for video_name in video_names:
        #     sample_and_save_on_video(video_name, video_embedder, risk_estimator, features, train_dataloader, folder="autogen")


        results[skill_name] = (wrong_test, wrong_train, wrong_nodrop, wrong_nodrop2)

    return results    

if __name__ == '__main__':
    test_all_final_benchmarks()
