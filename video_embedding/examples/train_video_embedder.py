#!/usr/bin/env python3

import risk_estimation
import video_embedding
from video_embedding.utils import behaviour_trial_names, get_session, set_session
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder
import argparse

from risk_estimation.models.risk_estimation.risk_dataloader import RiskEstimationDataset
from risk_estimation.models.risk_estimation.frame_dropping import NoFrameDroppingPolicy, OnlyLabelledFramesDroppingPolicy
from risk_estimation.models.risk_estimation.risk_feature_extractor import VideoObservationsRiskAndSafeLabels, LatentObservationsRiskLabels, StampedLatentObservationsRiskLabels
from video_embedding.utils import visualize_labelled_video_frame_inline
from video_embedding.models.nerual_networks.autoencoder import LargeAutoencoder, Autoencoder, CustomResnetStage1, CustomResnetStage2, CustomResnetStage3, CustomResnetStage4, CustomResnetStage5

import torchvision

from video_embedding.models.nerual_networks.autoencoder import LargeAutoencoder, Autoencoder, CustomResnetStage1, CustomResnetStage2, CustomResnetStage3, CustomResnetStage4, CustomResnetStage5
def main(args):
    set_session(args.session)
    video_embedder = RiskyBehavioralVideoEmbedder(name=args.video[0], latent_dim=int(args.latent_dim), behaviours=args.behaviours, learning_rate=float(args.learning_rate), augmentation=args.augmentation, nn_model=LargeAutoencoder)
    video_embedder.load(args.video, validation_videos=args.validation_video)
    
    video_embedder.create_video(epoch = args.epoch, patience=args.patience)
    video_embedder.save_model()
    video_embedder.save_latent_trajectory()

def get_custom_dataset(video_embedder, videos):
    train_dataloader_images, test_dataloader_images = RiskEstimationDataset.load(
        video_names=videos,
        video_embedder=video_embedder,
        batch_size=video_embedder.batch_size,
        frame_dropping_policy=OnlyLabelledFramesDroppingPolicy,
        features=VideoObservationsRiskAndSafeLabels,
    )
    X_test_images, Y_test_images = RiskEstimationDataset.dataloader_to_array(test_dataloader_images)
    X_train_images, Y_train_images = RiskEstimationDataset.dataloader_to_array(train_dataloader_images)
    return X_train_images

def plot_photos(video_embedder, video):
    X_train_images = get_custom_dataset([video])
    img_enc = video_embedder.model.forward(X_train_images).detach().cpu().numpy()
    print(img_enc.shape)
    visualize_labelled_video_frame_inline(img_enc[0], risk_flag=0, safe_flag=0, novelty_flag=0, recovery_phase=-1.0, press_for_next_frame=False, printer=False)
    visualize_labelled_video_frame_inline(img_enc[5], risk_flag=0, safe_flag=0, novelty_flag=0, recovery_phase=-1.0, press_for_next_frame=False, printer=False)
    visualize_labelled_video_frame_inline(img_enc[10], risk_flag=0, safe_flag=0, novelty_flag=0, recovery_phase=-1.0, press_for_next_frame=False, printer=False)

def update(args, plot: bool = False):
    assert len(args.video) == 1, "Put update videos to --video_updates"
    assert len(args.update_videos) > 0, "No --video_updates videos"
    assert args.behaviours is None
    set_session(args.session)

    video_embedder = RiskyBehavioralVideoEmbedder(name=args.video[0], latent_dim=int(args.latent_dim), behaviours=None, frame_dropping=True, learning_rate=float(args.learning_rate))
    video_embedder.load_model()

    print("video_embedder.model_train_record")
    print(video_embedder.model_train_record)

    if plot:
        plot_photos(video_embedder, args.video[0])
        plot_photos(video_embedder, args.update_videos[0])
    
    video_embedder.name = args.update_videos[0]
    all_videos = args.video + args.update_videos
    video_embedder.load(all_videos, validation_videos=args.validation_video)
    video_embedder.create_video(epoch=500)

    if plot:
        plot_photos(video_embedder, args.video[0])
        plot_photos(video_embedder, args.update_videos[0])

    video_embedder.save_model()
    video_embedder.save_latent_trajectory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Embedds video to latent space via AE",
        description="",
        epilog="",
    )

    parser.add_argument(
        "--video", nargs="+",
        default=["peg_door", "peg_door_trial_0", "peg_door_trial_1", "peg_door_trial_2", "peg_door_trial_3", "peg_door_trial_4", "peg_door_trial_5", "peg_door_trial_6"],
        help="put video name or video names for video embedder to be trained on"
    )
    parser.add_argument(
        "--validation_video", nargs="+",
        default=[],
        help="videos used to stop traning (validation_videos -> validation_dataloader -> embedder training gets loss every epoch -> based on loss training stops)",
    )
    parser.add_argument(
        "--session",
        default="",
    )
    parser.add_argument(
        "--behaviours", nargs="+",
        default=None, #["successful", "door"],
        help=""
    )
    parser.add_argument(
        "--latent_dim",
        default=16,
    )
    parser.add_argument("-lr",
        "--learning_rate",
        default=0.01,
    )
    parser.add_argument(
        "--epoch",
        default=1200,
        type=int,
    )
    parser.add_argument(
        "--patience",
        default=1200,
        type=int,
    )


    parser.add_argument(
        "--update_videos", nargs="+",
        default=[],
    )
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--no_update", dest="update", action="store_false")
    parser.set_defaults(update=False)

    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--no_augmentation", dest="augmentation", action="store_false")
    parser.set_defaults(augmentation=True)


    args = parser.parse_args()
    if args.update:
        update(args)
    else:
        main(args)
