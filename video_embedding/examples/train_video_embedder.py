#!/usr/bin/env python3

from video_embedding.utils import get_session, set_session, get_all_names
from video_embedding.models.video_embedder import VideoEmbedder
import argparse

from video_embedding.models.nerual_networks.autoencoder import *

def main(args):
    set_session(args.session)
    video_embedder = VideoEmbedder(
        name=args.video[0],
        latent_dim=int(args.latent_dim),
        learning_rate=float(0.001),
        batch_size=128,
        augmentation=False,
        nn_model=Autoencoder2,
    )

    all_videos = get_all_names(args.video)

    video_embedder.load(all_videos)

    video_embedder.train(num_epochs=600)
    # video_embedder.train(num_epochs=args.epoch, patience=args.patience)

    # video_embedder.create_video()
    video_embedder.save_model()
    # video_embedder.save_latent_trajectory()

def update(args, plot: bool = False):
    assert len(args.video) == 1, "Put update videos to --video_updates"
    assert len(args.update_videos) > 0, "No --video_updates videos"
    set_session(args.session)

    video_embedder = VideoEmbedder(
        name=args.video[0],
        latent_dim=int(args.latent_dim),
        frame_dropping=True,
        learning_rate=float(args.learning_rate),
    )
    video_embedder.load_model()

    print("video_embedder.model_train_record")
    print(video_embedder.model_train_record)


    video_embedder.name = args.update_videos[0]
    all_videos = args.video + args.update_videos
    video_embedder.load(all_videos)
    video_embedder.create_video(epoch=500)


    video_embedder.save_model()
    video_embedder.save_latent_trajectory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Embedds video to latent space via AE",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="peg_pick404",
        help="put video name or video names for video embedder to be trained on",
    )
    parser.add_argument(
        "--session",
        default="quantitative_study",
    )
    
    args = parser.parse_args()
    if args.update:
        update(args)
    else:
        main(args)
