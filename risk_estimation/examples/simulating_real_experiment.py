"""Making simulated dataset on labelled one + training
"""

import argparse
import risk_estimation
from risk_estimation.examples.train_markovian_risk_classifier import get_risk_estimator
from risk_estimation.models.risk_estimation.frame_dropping import ProactiveRiskLabelingDroppingPolicy, OnlyLabelledFramesDroppingPolicy, NoFrameDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import VideoObservationsRiskAndSafeLabels, StampedLatentObservationsRiskLabels, LatentObservationsRiskLabels, StampedDistLatentObservationsRiskLabels
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import all_trial_names, behaviour_trial_names, set_session, visualize_labelled_video
from pretty_confusion_matrix import pp_matrix_from_data
import video_embedding

def make_dataset_iteration_number(n, video_names, video_embedder, features):
    """Get the dataset that has n interactions with the user

    Args:
        n (_type_): _description_
        video_names (_type_): _description_
        video_embedder (_type_): _description_
        features (_type_): _description_

    Returns:
        _type_: _description_
    """    
    ProactiveRiskLabelingDroppingPolicy.limit_interest = n

    train_dataloader, test_dataloader = RiskEstimationDataset.load(
        video_names=video_names,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=ProactiveRiskLabelingDroppingPolicy,
        features=eval(features),
    )
    train_dataloader_images, test_dataloader_images = RiskEstimationDataset.load(
        video_names=video_names,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=ProactiveRiskLabelingDroppingPolicy,
        features=VideoObservationsRiskAndSafeLabels,
    )
    X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(test_dataloader_images)

    return train_dataloader, test_dataloader, X_test_images, Y_test_images



def main(args):
    if args.session != "":
        set_session(args.session)

    video_embedder = VideoEmbedder(latent_dim=args.video_latent_dim)
    video_embedder.load(args.skill_name)
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )

    risk_estimator = get_risk_estimator(args, video_embedder)

    # Risk Estimator loads data for training
    if args.behaviours == "":
        video_names = all_trial_names(args.skill_name, include_repr=True)
    else:
        video_names = behaviour_trial_names(args.skill_name, args.behaviours)


    for n in range(15,25,3):
        print(f"iterations: {n}")
        train_dataloader, test_dataloader, X_test_images, Y_test_images = make_dataset_iteration_number(n, video_names, video_embedder, features=args.features)

        print(f"Train with trials: {video_names}")
        risk_estimator.training_loop(train_dataloader)


        # Test 1
        X_test, Y_test = RiskEstimationDataset.dataloader_to_array(test_dataloader)
        pred, _ = risk_estimator.sample(X_test)
        Y_test = Y_test.cpu().numpy().squeeze()
        print(f"Test data samples: {len(Y_test)}, Risky: {len(Y_test[Y_test==1])}, Safe: {len(Y_test[Y_test==0])}")

        print(f"Accuracy on test data: {100 * (Y_test == pred).mean()}")
        if args.confusion_matrix:
            pp_matrix_from_data(Y_test, pred, columns=["Safe", "Danger"])

        import numpy as np
        indxs = np.where(Y_test != pred)
        visualize_labelled_video(X_test_images[indxs].cpu().numpy(), labels = {
            'risk_flag': Y_test_images.cpu().numpy()[indxs,0],
            'safe_flag': Y_test_images.cpu().numpy()[indxs,1],
        }, press_for_next_frame=True, printer=True)

        # Test 2
        X_train, Y_train = RiskEstimationDataset.dataloader_to_array(train_dataloader)
        pred, _ = risk_estimator.sample(X_train)
        Y_train = Y_train.cpu().numpy().squeeze()
        print(f"Train data samples: {len(Y_train)}, Risky: {len(Y_train[Y_train==1])}, Safe: {len(Y_train[Y_train==0])}")

        print(f"Accuracy on train data: {100 * (Y_train == pred).mean()}")
        if args.confusion_matrix:
            pp_matrix_from_data(Y_train, pred, columns=["Safe", "Danger"])

        # Test 3
        dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
            features=eval(args.features),)
        Y = dataset.Y.cpu().numpy().squeeze()

        Y_pred, _ = risk_estimator.sample(dataset.X)
        print(f"Train data samples: {len(Y_pred)}, Risky: {len(Y_pred[Y_pred==1])}, Safe: {len(Y_pred[Y_pred==0])}")
        
        print(f"Only Valuable framedrop: Accuracy on all data: {100 * (Y == Y_pred).mean()}%")
        if args.confusion_matrix:
            pp_matrix_from_data(Y, Y_pred, columns=["Safe", "Danger"])


        # Test 3
        dataset = RiskEstimationDataset.load_dataset(video_names, video_embedder,
            frame_dropping_policy=NoFrameDroppingPolicy,
            features=eval(args.features),)
        Y = dataset.Y.cpu().numpy().squeeze()

        Y_pred, _ = risk_estimator.sample(dataset.X)
        print(f"Train data samples: {len(Y_pred)}, Risky: {len(Y_pred[Y_pred==1])}, Safe: {len(Y_pred[Y_pred==0])}")
        
        print(f"No framedrop: Accuracy on all data: {100 * (Y == Y_pred).mean()}%")
        if args.confusion_matrix:
            pp_matrix_from_data(Y, Y_pred, columns=["Safe", "Danger"])





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Custom Train and Evaluate Risk Estimator",
        description="",
        epilog="",
    )
    parser.add_argument("-n", "--skill_name", default="peg_pick")
    parser.add_argument("-s", "--session", default="", help="Use subdirectory")
    parser.add_argument("-b", "--behaviours", nargs="+", default=['successful','cables', 'hands'])

    parser.add_argument("-l", "--video_latent_dim", default=8)
    parser.add_argument("-e", "--epoch", default=800, type=int)
    parser.add_argument(
        "-a",
        "--approach",
        default="GP",
        choices=["MLP", "DistLS", "GP", "LR", "DistLR", "DistMin"],
        help="Risk Estimator methods include MultiLayer Perceptron (MLP), Logistic Regression (LR), Gaussian Process (GP), Simiarity Distance with Linear Hyperparameter search (DistLS), Dist with Linear Regression search (DistLR), and Dist with cautious threshold applied (DistMin).",
    )

    parser.add_argument("--confusion_matrix", action="store_true")
    parser.add_argument("--no_confusion_matrix", dest="confusion_matrix", action="store_false")
    parser.set_defaults(confusion_matrix=True)
    
    parser.add_argument("--features",
        default="StampedLatentObservationsRiskLabels",
        choices=["LatentObservationsRiskLabels", # h
                 "StampedLatentObservationsRiskLabels", # h + alpha
                 "StampedDistLatentObservationsRiskLabels"], # h + alpha + d
        help="Latent (h), Stamped (alpha), Dist (d) features.",
    )

    main(parser.parse_args())