import json
from models.risk_estimation.result_evaluator import ResultEvaluator
import risk_estimation
from risk_estimation.plot_utils import plot_threshold_labelled
import video_embedding
from video_embedding.utils import all_trial_names, behaviour_trial_names, set_session, tensor_image_to_cv2, visualize_labelled_video
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder, VideoEmbedder
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, StampedVideoObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, VideoObservationsRiskLabels, StampedDistLatentObservationsRiskLabels
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


def test_load_specific_model(latent_dim=16):
    set_session("manipulation_demo_session")

    tuples = []
    tuples += [('peg_pick', ['successful', 'hands'])]
    tuples += [('peg_door', ['successful', 'hands'])]
    tuples += [('peg_place', ['successful', 'hands'])]
    tuples += [('peg_door', ['successful', 'door'])]
    tuples += [('peg_pick', ['successful', 'peg_rotated'])]
    tuples += [('peg_place', ['successful', 'peg_rotated'])]
    tuples += [('peg_pick',['successful', 'cables'])]
    tuples += [('peg_door',['successful', 'cables'])]
    tuples += [('peg_place',['successful', 'cables'])]
    
    for skill,behaviours in tuples:
        video_embedder = RiskyBehavioralVideoEmbedder(name=skill, latent_dim=latent_dim, behaviours=behaviours)
        video_embedder.load()
        video_embedder.load_model()

if __name__ == '__main__':
    test_load_specific_model()