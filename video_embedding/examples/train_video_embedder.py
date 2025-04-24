#!/usr/bin/env python3
from video_embedding.utils import set_session, get_all_names
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.models.video_embedding_dataset import load_dataloader
import argparse

def main(args):
    set_session(args.session)
    video_embedder = VideoEmbedder(
        name=args.video,
        latent_dim=12,
        learning_rate=0.001,
        nn_model=args.nn_model,
    )
    if args.update: video_embedder.load_model()

    train_videos = get_all_names(args.video)
    print(f"Training on: {train_videos}")
    dataloader = load_dataloader(train_videos, batch_size=64)

    video_embedder.train(dataloader, train_videos, num_epochs=args.num_epochs)

    video_embedder.save_model()
    video_embedder.create_video(dataloader)
    latent_traj = video_embedder.latent_trajectory(dataloader)
    video_embedder.visualize_latent_trajectory(latent_traj)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="peg_door404")
    parser.add_argument("--session", default="AE3")
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--nn_model", default="Autoencoder3")
    parser.add_argument("--update", action="store_true")
    parser.set_defaults(update=False)

    main(parser.parse_args())
