#!/usr/bin/env python3

from video_embedding.utils import get_session, set_session, get_all_names
from video_embedding.models.video_embedder import VideoEmbedder
import argparse

from video_embedding.models.nerual_networks.autoencoder import *

def main(args):
    set_session(args.session)
    video_embedder = VideoEmbedder(
        name=args.video,
        latent_dim=12,
        learning_rate=0.001,
        batch_size=128,
        nn_model=eval(args.nn_model),
    )

    all_videos = get_all_names(args.video)
    print(f"Training on: {all_videos}")
    video_embedder.load(all_videos)

    video_embedder.train(num_epochs=args.num_epochs)

    video_embedder.save_model()
    video_embedder.create_video()
    # video_embedder.save_latent_trajectory()

def update(args):
    set_session(args.session)

    video_embedder = VideoEmbedder(
        name=args.video,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="peg_pick404",
        help="put skill name",
    )
    parser.add_argument(
        "--session",
        default="quantitative_study",
    )
    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--nn_model",
        default="Autoencoder2",
    )
    
    args = parser.parse_args()
    # update(args)
    main(args)
