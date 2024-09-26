#!/usr/bin/env python3
import json
from models.risk_estimation.result_evaluator import ResultEvaluator
import risk_estimation
from risk_estimation.plot_utils import plot_threshold_labelled
import video_embedding
from video_embedding.utils import all_trial_names, behaviour_trial_names, set_session, tensor_image_to_cv2, visualize_labelled_video
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder, VideoEmbedder
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledPhaseDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsPhaseLabels, StampedVideoObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, VideoObservationsRiskLabels
from risk_estimation.models.risk_estimator import sample_and_save_on_video

from models.recovery_state_finder import GPRecoveryStateFinder

import argparse
from pretty_confusion_matrix import pp_matrix_from_data

def main(args):
    if args.session != "":
        set_session(args.session)

    video_embedder = RiskyBehavioralVideoEmbedder(
        name=args.skill_name,
        latent_dim=args.video_latent_dim,
    )
    video_embedder.load_model()

    recovery_estimator = GPRecoveryStateFinder(name=args.skill_name, xdim=args.video_latent_dim, learning_rate=0.01, train_patience=300)  

    
    video_names = all_trial_names(args.skill_name, include_repr=True)
    
    train_dataloader, test_dataloader = RiskEstimationDataset.load(
        video_names=video_names,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=eval(args.framedrop_policy),
        features=eval(args.features),
    )
    train_dataloader_images, test_dataloader_images = RiskEstimationDataset.load(
        video_names=video_names,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=eval(args.framedrop_policy),
        features=VideoObservationsRiskAndSafeLabels,
    )
    X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(test_dataloader_images)
    X_train_images, Y_train_images = RiskEstimationDataset.dataloader_to_array(train_dataloader_images)

    from torch.utils.data import DataLoader
    dataset_nodrop = RiskEstimationDataset.load_dataset(video_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=eval(args.features),)
    dataloader_nodrop = DataLoader(dataset_nodrop, batch_size=40, shuffle=False)

    if args.train:
        print(f"Train with trials: {video_names}")
        recovery_estimator.dataloader_test_for_plot = test_dataloader
        recovery_estimator.dataloader_nodrop_for_plot = dataloader_nodrop
        recovery_estimator.training_loop(train_dataloader, num_epochs=1000)
        if args.save:
            recovery_estimator.save_model()
    else:
        recovery_estimator.load_model()

    print("Test")
    X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dataloader)
    Y_test = Y_test.cpu().numpy().squeeze()
    Y_pred, _ = recovery_estimator.sample(X_test)
    print("Accuracy: ", ((Y_test - Y_pred) < 1).mean())
    clmns = [f"a={i}" for i in range(10)]
    pp_matrix_from_data(Y_test.round(), Y_pred.round(), columns=clmns, name="Test dataset", savepath=None)

    plot_where_wrong(video_embedder, Y_test, Y_pred, X_test_images, Y_test_images)

    print("Train")
    X_train, Y_train = RiskEstimationDataset.dataloader_to_array(train_dataloader)
    Y_train = Y_train.cpu().numpy().squeeze()
    Y_pred, _ = recovery_estimator.sample(X_train)
    print("Accuracy: ", ((Y_train - Y_pred) < 1).mean())
    pp_matrix_from_data(Y_test.round(), Y_pred.round(), columns=clmns, name="Train dataset", savepath=None)


    # e = ResultEvaluator(name="Test dataset", iwanttosee=["accuracy"])
    # e(recovery_estimator, video_embedder, X_test, Y_test, X_test_images, Y_test_images)



    # e = ResultEvaluator(name="Train dataset", iwanttosee=["accuracy"])
    # e(recovery_estimator, video_embedder, X_train, Y_train, X_train_images, Y_train_images)

    # dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
    #     frame_dropping_policy=NoFrameDroppingPolicy,
    #     features=eval(args.features),)
    # dataset_images = RiskEstimationDataset.load_dataset(video_names, video_embedder,
    #     frame_dropping_policy=NoFrameDroppingPolicy,
    #     features=VideoObservationsRiskAndSafeLabels,)
    # e = ResultEvaluator(name="NoDrop, Prior is Safe", iwanttosee=["accuracy"])
    # e(recovery_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)

    # dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
    #     frame_dropping_policy=NoFrameDroppingPolicy,
    #     features=eval(args.features+"PriorRisk"),)
    # dataset_images = RiskEstimationDataset.load_dataset(video_names, video_embedder,
    #     frame_dropping_policy=NoFrameDroppingPolicy,
    #     features=VideoObservationsRiskAndSafeLabels,)
    # e = ResultEvaluator(name="NoDrop, Prior is Risk", iwanttosee=["accuracy"])
    # e(recovery_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)

    # if args.save_videos:
    #     for video_name in video_names:
    #         sample_and_save_on_video(video_name, video_embedder, recovery_estimator, args.features, train_dataloader)

import numpy as np
def plot_where_wrong(video_embedder, Y_test, Y_pred, X_test_images, Y_test_images):
    indxs = np.where(Y_test != Y_pred)[0]

    decoded_images = []
    for idx in indxs:
        latent = video_embedder.model.encoder(X_test_images[idx:idx+1])
        decoded_img = video_embedder.model.decoder(latent)
        decoded_images.append(decoded_img.detach().cpu().numpy())
        
    print("now decoded images")
    x = decoded_images
    y = Y_test_images.cpu().numpy()[indxs]

    labels = {}
    print(y)
    print(y.shape)
    if len(y) > 0:
        labels = {
            'risk_flag': y[:,0],
            'safe_flag': y[:,1],
            'recovery_phase': Y_pred[indxs],
        }
    
    visualize_labelled_video(x, labels, press_for_next_frame=True, printer=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Custom Train and Evaluate Risk Estimator",
        description="",
        epilog="",
    )
    
    parser.add_argument("--skill_name", default="peg_door")
    parser.add_argument("--session", default="recovery_session", help="Uses subdirectory")

    parser.add_argument("--video_latent_dim", default=16, help="Chooses video embedding model with dim", type=int)
    parser.add_argument("--approach", default="GP")

    parser.add_argument("--features",
        default="LatentObservationsPhaseLabels",
        choices=["LatentObservationsPhaseLabels", # h
                 "StampedLatentObservationsPhaseLabels", # h + alpha
                 "StampedDistLatentObservationsPhaseLabels"], # h + alpha + d
        help="Latent (h), Stamped (alpha), Dist (d) features.",
    )
    parser.add_argument("--framedrop_policy",
        default="NoFrameDroppingPolicy",
        choices=["OnlyLabelledPhaseDroppingPolicy","NoFrameDroppingPolicy"],
    )

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.set_defaults(train=True)
    
    parser.add_argument("--save", action="store_true", help="Saves model to ./saved_models/<session>/<skill>_<approach>_<xdim>_model.pt")
    parser.add_argument("--no_save", dest="save", action="store_false")
    parser.set_defaults(save=False)

    parser.add_argument("--save_videos", action="store_true", help="Saves processed videos to ./videos/<session>/<skill>/<skill_trial>/<skill video>.mp4 and risk estimated data to ./videos/<skill>/<skill_trial>/<risk estimated data>.csv")
    parser.add_argument("--no_save_videos", dest="save_videos", action="store_false")
    parser.set_defaults(save_videos=False)

    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--no_confusion_matrix", dest="confusion_matrix", action="store_false")
    parser.set_defaults(confusion_matrix=True)


    main(parser.parse_args())
