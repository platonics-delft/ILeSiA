import numpy as np
import torch

from video_embedding.utils import get_session, load, set_session, visualize_labelled_video, visulize_video, load_video
from video_embedding.models.video_embedder import RiskyBehavioralVideoEmbedder
import video_embedding
import argparse

def main(args):

    set_session(args.session)
    video_embedder = RiskyBehavioralVideoEmbedder(name=args.video,latent_dim=args.latent_dim)
    images=load_video(name=args.video)

    video_embedder.load_model()
    # video_embedder.visulize_video(images)

    decoded_images = []
    for image in images:
        tensor_image = torch.tensor(image, dtype=torch.float32)
        tensor_image= tensor_image.cuda()
        tensor_image=tensor_image.unsqueeze(0).unsqueeze(0)

        decoded_image = video_embedder.model.forward(tensor_image).cpu().detach().numpy()[0][0]
        decoded_images.append(decoded_image)

    visualize_labelled_video(decoded_images, press_for_next_frame=True, printer=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Play video and latent space",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--video",
        default="peg_door_trial_2",
    )
    parser.add_argument(
        "--session",
        default="",
    )
    parser.add_argument(
        "--latent_dim",
        default=16,
        type=int,
    )
    
    main(parser.parse_args())
