import numpy as np
from risk_estimation.models.safety_layer import SafetyLayer
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, StampedLatentObservationsRiskLabels
from risk_estimation.models.risk_estimator import (
    DistanceRiskEstimator,
    NMDistanceRiskEstimatorDTW,
    MLPRiskEstimator,
)
import torch
import video_embedding, risk_estimation
from video_embedding.models.video_embedder import VideoEmbedder, RiskyBehavioralVideoEmbedder

from scipy.spatial.distance import cosine, euclidean
from video_embedding.utils import all_trial_names, set_session


def test_main_video_embedder(video = "peg_door", behaviours = []):

    video_embedder = RiskyBehavioralVideoEmbedder(name=video, latent_dim=16, behaviours=behaviours)
    
    video_embedder.load()
    video_embedder.create_video(epoch=1)
    # video_embedder.save_model()
    # video_embedder.save_latent_trajectory()




def test_risk_estimator_mlp(features=LatentObservationsRiskLabels):
    set_session("test_session")
    latent_dim = 8
    skill_name = 'peg_door'

    video_embedder = VideoEmbedder(latent_dim)
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/", name=skill_name
    )

    risk_estimator = MLPRiskEstimator(
        skill_name,
        features.xdim(latent_dim),
        video_embedder.batch_size,
    )
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names=all_trial_names("peg_door"),
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
        features=features,
    )

    risk_estimator.training_loop(train_dl, num_epochs=1500)
    
    X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dl)

    Y_test = Y_test.cpu().numpy().squeeze()
    Y_pred, _ = risk_estimator.sample(X_test)

    acc = 100 * (Y_test == Y_pred).mean()
    return acc

def test_risk_estimator_mlp_accuracy_test():
    acc = test_risk_estimator_mlp()
    assert acc > 90, f"Accuracy on test data should be above 90%, it is {acc}"

def test_risk_estimator_mlp_accuracy_stamped():
    acc = test_risk_estimator_mlp(features=StampedLatentObservationsRiskLabels)
    assert acc > 96, f"Accuracy on test data should be above 96%, it is {acc}"

def test_risk_estimator_mlp_accuracy_stampeddist():
    acc = test_risk_estimator_mlp(features=StampedDistLatentObservationsRiskLabels)
    assert acc > 96, f"Accuracy on test data should be above 96%, it is {acc}"

def test_risk_estimator_euclidean_distance():
    set_session("test_session")
    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.load("peg_door")  # load data

    # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    risk_estimator = DistanceRiskEstimator("peg_door_trial_0", dist_fun=euclidean, thr=10)
    
    risk_estimator.load_representation("peg_door_trial_0", video_embedder)
    # Test on all data
    test, pred = risk_estimator.test_all_on_video_names(
        ["peg_door_trial_1",  "peg_door_trial_2", "peg_door_trial_3", "peg_door_trial_4"],
        video_embedder,
    )
    # TODO: Use video_embedder.re.test_dataloader

    acc = 100 * (test == pred).mean()
    print(acc)
    return acc

def test_risk_estimator_cosine_distance():
    set_session("test_session")
    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.load("peg_door")  # load data

    # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    risk_estimator = DistanceRiskEstimator("peg_door_trial_0", dist_fun=cosine, thr=1.5)

    risk_estimator.load_representation("peg_door_trial_0", video_embedder)
    # Test on all data
    test, pred = risk_estimator.test_all_on_video_names(
        ["peg_door_trial_0", "peg_door_trial_1",  "peg_door_trial_2", "peg_door_trial_3"],
        video_embedder,
    )
    # TODO: Use video_embedder.re.test_dataloader

    acc = 100 * (test == pred).mean()
    return acc

def test_risk_estimator_cosine_distance_accuracy_test():
    acc = test_risk_estimator_cosine_distance()
    print(f"Accuracy: {acc}")
    assert acc > 60, f"Accuracy on test data should be above 60%, it is {acc}"

def test_risk_estimator_euclidean_distance_dtw():
    set_session("test_session")
    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.load("peg_door")  # load data

    # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    risk_estimator = NMDistanceRiskEstimatorDTW(
        "peg_door_trial_0", dist_fun=euclidean, thr=10)

    risk_estimator.load_representation("peg_door_trial_0", video_embedder)
    # Test on all data
    test, pred = risk_estimator.test_all_on_video_names(
        ["peg_door_trial_0", "peg_door_trial_1",  "peg_door_trial_2", "peg_door_trial_3"],
        video_embedder,
    )
    # TODO: Use video_embedder.re.test_dataloader

    acc = 100 * (test == pred).mean()
    return acc

def test_risk_estimator_euclidean_distance_dtw_accuracy_test():
    acc = test_risk_estimator_euclidean_distance_dtw()
    print(f"Accuracy: {acc}")
    assert acc > 65, f"Accuracy on test data should be above 65%, it is {acc}"

def test_risk_estimator_cosine_distance_dtw():
    set_session("test_session")
    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim=8)
    video_embedder.load("peg_door")  # load data

    # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    risk_estimator = NMDistanceRiskEstimatorDTW(
        "peg_door_trial_0", dist_fun=cosine, thr=1.5)

    risk_estimator.load_representation("peg_door_trial_0", video_embedder)
    # Test on all data
    test, pred = risk_estimator.test_all_on_video_names(
        ["peg_door_trial_0", "peg_door_trial_1",  "peg_door_trial_2", "peg_door_trial_3"],
        video_embedder,
    )
    # TODO: Use video_embedder.re.test_dataloader

    acc = 100 * (test == pred).mean()
    return acc

def test_risk_estimator_cosine_distance_dtw_accuracy_test():
    acc = test_risk_estimator_cosine_distance_dtw()
    print(f"Accuracy: {acc}")
    assert acc > 60, f"Accuracy on test data should be above 60%, it is {acc}"





def test_deplyed_model_usage():

    set_session("my_new_session")
    sl = SafetyLayer()

    # self.get_observations()
    observations = [
        torch.tensor(np.zeros((1,1,64,64)), dtype=torch.float32).cuda(), # 1. Image
        None, # 2. Risk Label flag
        None, # 3. Safe Label flag
        None, # 4. Novelty Label flag
        torch.tensor(np.zeros((1,1)), dtype=torch.float32).cuda() # 5. Frame number normalized (0-1)
    ]

    risk_pred = sl.estimate_risk(observations)

    tensor_images, risk_flag, safe_flag, novelty_flag, frame_number, recovery_phase = RiskEstimationDataset.load_video_data(name='peg_door')
    o = [
        tensor_images[0:1],
        None,
        None,
        None,
        frame_number[0:1],
    ]

    risk_pred = sl.estimate_risk(o)

    sl.update()
    sl.update_video_embedding(epoch=1)

def test_perfect_door_session():

    set_session("perfect_door_session")
    sl = SafetyLayer()
    sl.update()
    # sl.update_video_embedding(epoch=1)



if __name__ == "__main__":
    pass
    # test_risk_estimator_mlp()
    # test_risk_estimator_mlp_accuracy_test()
    # test_risk_estimator_mlp_accuracy_stamped()
    # test_risk_estimator_euclidean_distance()
    # test_risk_estimator_euclidean_distance_accuracy_test()
    # test_risk_estimator_cosine_distance()
    # test_risk_estimator_cosine_distance_accuracy_test()
    # test_risk_estimator_euclidean_distance_dtw()
    # test_risk_estimator_euclidean_distance_dtw_accuracy_test()
    # test_risk_estimator_cosine_distance_dtw()
    # test_risk_estimator_cosine_distance_dtw_accuracy_test()
    test_deplyed_model_usage()

    # test_perfect_door_session()