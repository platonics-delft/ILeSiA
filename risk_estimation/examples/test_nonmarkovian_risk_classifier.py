import numpy as np
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.risk_feature_extractor import StampedLatentObservationsRiskLabels
from risk_estimation.models.risk_estimator import (
    DistanceRiskEstimator,
    NMDistanceRiskEstimatorDTW,
    MLPRiskEstimator,
    NMDistanceRiskEstimator,
)
import video_embedding
from video_embedding.models.video_embedder import VideoEmbedder
import risk_estimation
import argparse
from pretty_confusion_matrix import pp_matrix_from_data

from scipy.spatial.distance import cosine, euclidean
from video_embedding.utils import all_trial_names

def main(args):
    # Video embedder encodes skill from video
    video_embedder = VideoEmbedder(latent_dim=args.video_latent_dim)
    video_embedder.load(args.skill_name)  # load data
    video_embedder.load_model(
        path=video_embedding.path + "/saved_models/"
    )  # OR load model

    if args.approach == 'DTW':
        risk_estimator = NMDistanceRiskEstimatorDTW(name=args.skill_name,dist_fun=eval(args.distance_fun), thr=args.threshold, video_embedder=video_embedder)
    else:
        risk_estimator = NMDistanceRiskEstimator(name=args.skill_name, dist_fun=eval(args.distance_fun), thr=args.threshold, video_embedder=video_embedder)


    ## Train hyperparams
    # Risk Estimator loads data for training
    # video_names = all_trial_names(args.skill_name, include_repr=True)
    # train_dataloader, test_dataloader = RiskEstimationDataset.load(
    #     video_names=video_names,
    #     video_embedder=video_embedder,
    #     batch_size=video_embedder.batch_size,
    #     frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
    #     features=StampedLatentObservationsRiskLabels,
    # )

    # if args.train:
    #     print(f"Train with trials: {video_names}")
    #     risk_estimator.training_loop(
    #         train_dataloader, num_epochs=1000, patience=10000
    #     )  # Train Risk Aware module
    #     if args.save:
    #         risk_estimator.save_model(
    #             path=risk_estimator.path + "/saved_models/"
    #         )
    # else:
    # risk_estimator.load_model(
    #     path=risk_estimator.path + "/saved_models/",
    # )  # OR load model


    ## Dataset of trajectories collection
    video_names = all_trial_names(args.skill_name, include_repr=True)
    trajectories = []
    label_trajectories = []
    for video_name in video_names:
        # What dropping and what features?
        dataset_traj = RiskEstimationDataset.load_dataset([video_name], video_embedder, frame_dropping_policy=NoFrameDroppingPolicy,
            features=StampedLatentObservationsRiskLabels)
        
        trajectory = dataset_traj.X.cpu().detach().numpy()
        label_trajectory = dataset_traj.Y.cpu().detach().numpy().squeeze()

        ## discard (frame number of other features)
        trajectory = trajectory[:,:video_embedder.latent_dim]

        trajectories.append(trajectory)
        label_trajectories.append(label_trajectory)
    
    trajectories = np.array(trajectories, dtype=object)
    label_trajectories = np.array(label_trajectories, dtype=object)
    ## predict
    print("Trajectories: ", trajectories.shape, trajectories[0].shape)
    print("Label trajectories: ", label_trajectories.shape, label_trajectories[0].shape)
    try: # DTW
        pred, risk, idxs = risk_estimator.sample(trajectories)
    except: # No DTW no need for idxs
        pred, risk = risk_estimator.sample(trajectories)
        idxs = None

    if idxs is not None:
        new_label_trajectories = []
        for n in range(len(label_trajectories)):
            label_trajectory = label_trajectories[n]
            # pick the indexed that were picked
            new_label_trajectories.append(label_trajectory[idxs[n]])

        label_trajectories = new_label_trajectories

    label_trajectories = np.concatenate(label_trajectories).ravel()
    pred = np.concatenate(pred).ravel()

    print("After label_trajectories: ", label_trajectories.shape)
    print("After pred: ", pred.shape)

    print(f"Accuracy on all (no dropping) data: {(label_trajectories == pred).mean()}")
    pp_matrix_from_data(label_trajectories, pred, columns=["Safe", "Danger"])

    ## Apply frame dropping policy
    ### Totally crazy extraction
    total_n = 0
    only_valuable_idxs = []
    for video_name in video_names:
        noframedrop_dataset = RiskEstimationDataset.load_dataset([video_name], video_embedder, frame_dropping_policy=NoFrameDroppingPolicy,
                features=StampedLatentObservationsRiskLabels)

        valuableframedrop_dataset = RiskEstimationDataset.load_dataset([video_name], video_embedder, frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
                features=StampedLatentObservationsRiskLabels)

        noframedrop_X = noframedrop_dataset.X.cpu().detach().numpy()
        valuableframedrop_X = valuableframedrop_dataset.X.cpu().detach().numpy()

        if len(valuableframedrop_X) == 0:
            continue

        print(noframedrop_X.shape)
        print(valuableframedrop_X.shape)

        # this should be all frame idxs 0,1,2,3,4,5,6,...,N
        for frame_n in noframedrop_X[:, 8]: # all frame idxs 
            if frame_n in valuableframedrop_X[:, 8]: # frame idxs
                only_valuable_idxs.append(total_n)

            total_n += 1
    
    label_trajectories = label_trajectories[only_valuable_idxs]
    pred = pred[only_valuable_idxs]
    print(f"Accuracy on all data: {(label_trajectories == pred).mean()}")
    pp_matrix_from_data(label_trajectories, pred, columns=["Safe", "Danger"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Risk Estimator ",
        description="Trains or Tests Risk Estimator module",
        epilog="",
    )
    parser.add_argument("-n", "--skill_name", default="peg_door")

    parser.add_argument("-l", "--video_latent_dim", default=8)
    parser.add_argument(
        "-a",
        "--approach",
        default="DTW",
        choices=["", "DTW"],
    )
    parser.add_argument(
        "-d",
        "--distance_fun",
        default="cosine",
        choices=["cosine", "euclidean"],
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.40,
        type=int,
    )

    # Train Hyperparams
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--no_train", dest="train", action="store_false")
    parser.set_defaults(train=True)

    
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no_save", dest="save", action="store_false")
    parser.set_defaults(save=False)

    main(parser.parse_args())
