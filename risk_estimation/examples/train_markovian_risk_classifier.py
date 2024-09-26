#!/usr/bin/env python3
import json
from models.risk_estimation.result_evaluator import ResultEvaluator
import risk_estimation
from risk_estimation.plot_utils import plot_threshold_labelled
import video_embedding
from video_embedding.utils import all_trial_names, all_test_names, behaviour_trial_names, set_session, tensor_image_to_cv2, visualize_labelled_video
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder, VideoEmbedder
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import LatentObservationsRiskLabels, StampedLatentObservationsRiskLabels, StampedVideoObservationsRiskLabels, VideoObservationsRiskAndSafeLabels, VideoObservationsRiskLabels, StampedDistLatentObservationsRiskLabels, LatentObservationsRiskLabelsPriorRisk, StampedLatentObservationsRiskLabelsPriorRisk, StampedDistLatentObservationsRiskLabelsPriorRisk
from risk_estimation.models.risk_estimator import (
    DistanceRiskEstimator,
    LinSearchDistanceRiskEstimator,
    NMDistanceRiskEstimatorDTW,
    GPRiskEstimator,
    LRHyperTrainDistanceRiskEstimator,
    MLPRiskEstimator,
    MinHyperTrainDistanceRiskEstimator,
    sample_and_save_on_video,
)
from risk_estimation.models.risk_estimation.benchmark_utils import (
    benchmark_eval_save,
)

from risk_estimation.models.safety_layer import get_risk_estimator_from_args

import argparse

def main(args, i = 0):
    if args.session != "":
        set_session(args.session)

    video_embedder = RiskyBehavioralVideoEmbedder(
        name=args.skill_name,
        latent_dim=args.video_latent_dim,
        behaviours=args.encoded_behaviours,
    )
    video_embedder.load_model()

    risk_estimator = get_risk_estimator_from_args(args, video_embedder)

    if args.behaviours is None:
        video_names = all_trial_names(args.skill_name, include_repr=True)
        test_video_names = all_test_names(args.skill_name)
    elif args.behaviours == "cross_validation":
        print("cross-validation experiment")
        video_names = all_trial_names(args.skill_name, include_repr=True)
        test_video_names = [video_names.pop(i)]
    else:
        video_names = behaviour_trial_names(args.skill_name, args.behaviours)
        test_video_names = all_test_names(args.skill_name)



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
    ## TESTING TO SWITCH TEST DATASET TO N-1 
    from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
    # test_video_names = all_test_names(args.skill_name)
    test_dataset = RiskEstimationDataset.load_dataset(test_video_names, video_embedder,
        frame_dropping_policy=eval(args.framedrop_policy),
        features=eval(args.features),)
    test_dataloader_2 = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)

    _, _, test_dataset, test_imgset = RiskEstimationDataset.extended_load(
        video_names, test_video_names, video_embedder, 
        eval(args.framedrop_policy), eval(args.features), resnet_option=False
    )
    test_dataloader_2 = DataLoader(test_dataset, batch_size=video_embedder.batch_size, shuffle=True)


    X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(test_dataloader_images)
    X_train_images, Y_train_images = RiskEstimationDataset.dataloader_to_array(train_dataloader_images)

    from torch.utils.data import DataLoader
    dataset_nodrop = RiskEstimationDataset.load_dataset(video_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=eval(args.features),)
    dataloader_nodrop = DataLoader(dataset_nodrop, batch_size=40, shuffle=False)

    if args.train:
        print(f"Train with trials: {video_names}")
        risk_estimator.dataloader_test_for_plot = test_dataloader_2
        risk_estimator.dataloader_nodrop_for_plot = dataloader_nodrop
        
        risk_estimator.training_loop(train_dataloader, early_stop=True)
        if args.save:
            risk_estimator.save_model()
    else:
        risk_estimator.load_model()


    X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dataloader)
    e = ResultEvaluator(name="Test dataset", iwanttosee=["accuracy", "confusion_matrix", 
"image_triplets"])
    e(risk_estimator, video_embedder, X_test, Y_test, X_test_images, Y_test_images)

    X_train, Y_train = RiskEstimationDataset.dataloader_to_array(train_dataloader)
    e = ResultEvaluator(name="Train dataset", iwanttosee=["accuracy", "confusion_matrix", 
"image_triplets"])
    e(risk_estimator, video_embedder, X_train, Y_train, X_train_images, Y_train_images)

    dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=eval(args.features),)
    dataset_images = RiskEstimationDataset.load_dataset(video_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=VideoObservationsRiskAndSafeLabels,)
    e = ResultEvaluator(name="NoDrop, Prior is Safe", iwanttosee=["accuracy", "confusion_matrix"])
    e(risk_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)

    dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=eval(args.features+"PriorRisk"),)
    dataset_images = RiskEstimationDataset.load_dataset(video_names, video_embedder,
        frame_dropping_policy=NoFrameDroppingPolicy,
        features=VideoObservationsRiskAndSafeLabels,)
    e = ResultEvaluator(name="NoDrop, Prior is Risk", iwanttosee=["accuracy", "confusion_matrix"])
    e(risk_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)

    # Generalization tests
    # dataset = RiskEstimationDataset.load_dataset(test_video_names, video_embedder,
    #     frame_dropping_policy=eval(args.framedrop_policy),
    #     features=eval(args.features),)
    # dataset_images = RiskEstimationDataset.load_dataset(test_video_names, video_embedder,
    #     frame_dropping_policy=eval(args.framedrop_policy),
    #     features=VideoObservationsRiskAndSafeLabels,)
    # e = ResultEvaluator(name="Generalizatrion test augmentation", iwanttosee=["accuracy", "confusion_matrix", "image_triplets"])
    # e(risk_estimator, video_embedder, dataset.X, dataset.Y, dataset_images.X, dataset_images.Y)


    
    benchmark_eval_save("Test_dataset", args.skill_name, test_dataset, test_imgset, video_embedder, risk_estimator)

    if args.save_videos:
        for video_name in video_names:
            sample_and_save_on_video(video_name, video_embedder, risk_estimator, args.features, train_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Custom Train and Evaluate Risk Estimator",
        description="",
        epilog="",
    )
    
    parser.add_argument("--skill_name", default="peg_door")
    parser.add_argument("--session", default="", help="Uses subdirectory")
    parser.add_argument("--train_epoch", default=1000)
    parser.add_argument("--train_patience", default=1000)

    parser.add_argument("--out_assessment", default='optimistic', choices=['optimistic', 'cautious'],
                        help="Optimistic Risk Assessment (mean) or Cautious Risk Assessment (mean + std)")


    parser.add_argument("--video_latent_dim", default=16, help="Chooses video embedding model with dim", type=int)
    parser.add_argument("--approach",
        default="GP",
        choices=["MLP", "DistLS", "GP", "LR", "DistLR", "DistMin", "L+GP", "L+GP+1SKIP", "L+GP+2SKIP"],
        help="Risk Estimator methods include MultiLayer Perceptron (MLP), Logistic Regression (LR), Gaussian Process (GP), Simiarity Distance with Linear Hyperparameter search (DistLS), Dist with Linear Regression search (DistLR), and Dist with cautious threshold applied (DistMin).",
    )

    parser.add_argument("--features",
        default="StampedLatentObservationsRiskLabels",
        choices=["LatentObservationsRiskLabels", # h
                 "StampedLatentObservationsRiskLabels", # h + alpha
                 "StampedDistLatentObservationsRiskLabels"], # h + alpha + d
        help="Latent (h), Stamped (alpha), Dist (d) features.",
    )
    parser.add_argument("--framedrop_policy",
        default="OnlyLabelledFramesDroppingPolicy",
        choices=["OnlyLabelledFramesDroppingPolicy",],
    )

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.set_defaults(train=True)
    
    parser.add_argument("--save", action="store_true", help="Saves model to ./saved_models/<session>/<skill>_<approach>_<xdim>_model.pt")
    parser.add_argument("--no_save", dest="save", action="store_false")
    parser.set_defaults(save=True)

    parser.add_argument("--save_videos", action="store_true", help="Saves processed videos to ./videos/<session>/<skill>/<skill_trial>/<skill video>.mp4 and risk estimated data to ./videos/<skill>/<skill_trial>/<risk estimated data>.csv")
    parser.add_argument("--no_save_videos", dest="save_videos", action="store_false")
    parser.set_defaults(save_videos=True)

    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--no_confusion_matrix", dest="confusion_matrix", action="store_false")
    parser.set_defaults(confusion_matrix=True)

    # Special experimental cases -> evaluate bades on descripted risky-behaviours
    parser.add_argument("--encoded_behaviours", default=None, help="Request encoded model with following risky-behaviours")
    parser.add_argument("--behaviours", nargs="+", default=None)

    main(parser.parse_args())
