import numpy as np
from risk_estimation.models.safety_layer import SafetyLayer
from risk_estimation.datasets.frame_dropping import *
from risk_estimation.datasets.risk_dataloader import RiskEstimationDataset
from risk_estimation.datasets.risk_feature_extractor import *
from risk_estimation.models.risk_estimator import *
import torch
import video_embedding
from video_embedding.models.video_embedder import VideoEmbedder

from scipy.spatial.distance import cosine, euclidean
from video_embedding.utils import all_trial_names, set_session

from risk_estimation.models.mlp_risk_estimator import MLPRiskEstimator
from risk_estimation.models.dist_risk_estimator import DistanceRiskEstimator

def test_main_video_embedder(video = "peg_door404"):
    set_session("AE3")
    video_embedder = VideoEmbedder(name=video, latent_dim=12)
    video_embedder.load(videos=[video])
    video_embedder.train(num_epochs=1)

def test_risk_estimator_mlp(features=LatentObservationsRiskLabels, skill_name='peg_door404'):
    set_session("AE3")

    video_embedder = VideoEmbedder(name=skill_name, latent_dim=12)
    video_embedder.load_model()

    risk_estimator = MLPRiskEstimator(
        skill_name,
        features.xdim(12),
        video_embedder.batch_size,
        train_epoch=1,
    )
    train_dl, test_dl = RiskEstimationDataset.load(
        video_names=all_trial_names("peg_door404"),
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
        features=features,
    )

    risk_estimator.training_loop(train_dl)
    
    X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dl)

    Y_test = Y_test.cpu().numpy().squeeze()
    Y_pred, _, _ = risk_estimator.sample(X_test)

    acc = 100 * (Y_test == Y_pred).mean()


# def test_risk_estimator_euclidean_distance(skill_name='peg_door404'):
#     set_session("AE3")
#     # Video embedder encodes skill from video
#     video_embedder = VideoEmbedder(name=skill_name, latent_dim=12)
#     video_embedder.load()  # load data

#     # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
#     video_embedder.load_model(
#         path=video_embedding.path + "/saved_models/"
#     )  # OR load model

#     risk_estimator = DistanceRiskEstimator("peg_door_trial_0", dist_fun=euclidean, thr=10)
    
#     risk_estimator.load_representation("peg_door_trial_0", video_embedder)
#     # Test on all data
#     test, pred = risk_estimator.test_all_on_video_names(
#         ["peg_door_trial_1",  "peg_door_trial_2", "peg_door_trial_3", "peg_door_trial_4"],
#         video_embedder,
#     )
#     # TODO: Use video_embedder.re.test_dataloader

#     acc = 100 * (test == pred).mean()
#     print(acc)
#     return acc

# def test_risk_estimator_cosine_distance():
#     set_session("test_session")
#     # Video embedder encodes skill from video
#     video_embedder = VideoEmbedder(latent_dim=8)
#     video_embedder.load("peg_door")  # load data

#     # video_embedder.create_video(path=risk_estimation.path + "/videos/")  # train ae
#     video_embedder.load_model(
#         path=video_embedding.path + "/saved_models/"
#     )  # OR load model

#     risk_estimator = DistanceRiskEstimator("peg_door_trial_0", dist_fun=cosine, thr=1.5)

#     risk_estimator.load_representation("peg_door_trial_0", video_embedder)
#     # Test on all data
#     test, pred = risk_estimator.test_all_on_video_names(
#         ["peg_door_trial_0", "peg_door_trial_1",  "peg_door_trial_2", "peg_door_trial_3"],
#         video_embedder,
#     )
#     # TODO: Use video_embedder.re.test_dataloader

#     acc = 100 * (test == pred).mean()
#     return acc

def test_deplyed_model_usage():

    set_session("AE3")
    sl = SafetyLayer(skill_name="peg_door404")

    # self.get_observations()
    observations = [
        torch.tensor(np.zeros((1,1,64,64)), dtype=torch.float32).cuda(), # 1. Image
        None, # 2. Risk Label flag
        None, # 3. Safe Label flag
        None, # 4. Novelty Label flag
        torch.tensor(np.zeros((1,1)), dtype=torch.float32).cuda() # 5. Frame number normalized (0-1)
    ]

    risk_pred = sl.estimate_risk(observations)

    tensor_images, risk_flag, safe_flag, novelty_flag, frame_number, recovery_phase = RiskEstimationDataset.load_video_data(name='peg_door404')
    o = [
        tensor_images[0:1],
        None,
        None,
        None,
        frame_number[0:1],
    ]

    risk_pred = sl.estimate_risk(o)
    print("DONE!")
    
    # sl.update()
    # sl.update_video_embedding(epoch=1)



if __name__ == "__main__":
    # test_deplyed_model_usage()
    pass
