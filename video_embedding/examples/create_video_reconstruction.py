


import PIL.Image
import torch
from video_embedding.models.nerual_networks.autoencoder import Autoencoder2
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import set_session
# import cv2
import PIL

def get_all_names():
    all_names: list[str] = []
    for i in range(30):
        all_names.append(f"peg_door404_test_{i}")
    for i in range(3):
        all_names.append(f"peg_door404_trial_{i}")
    all_names.append("peg_door404")
    return all_names

# load model src/ILeSiA/video_embedding/saved_models/quantitative_study/peg_place404_model_12.pt

set_session("quantitative_study")
video_embedder = VideoEmbedder(
        name="peg_door404",
        latent_dim=int(12),
        learning_rate=float(0.001),
        batch_size=120,
        augmentation=False,
        nn_model="Autoencoder2",
    )

video_embedder.load_model()
    # torch.load("src/ILeSiA/video_embedding/saved_models/quantitative_study/peg_place404_model_12.pt")

# nn_model = Autoencoder2(latent_dim=12)

# nn_model.load_state_dict(torch.load("src/ILeSiA/video_embedding/saved_models/quantitative_study/peg_place404_model_12.pt"))


# video_embedder.load(videos=["peg_place404"], validation_videos=[], shuffle=False)
video_embedder.load(videos=get_all_names(), shuffle=False)

data_iter = iter(video_embedder.dataloader)
batch0 = data_iter.__next__()

PIL.Image.fromarray((batch0[0].cpu().numpy()[0].squeeze(0))*255).show()


video_embedder.create_video()


