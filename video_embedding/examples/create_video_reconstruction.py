


import PIL.Image
import torch
from video_embedding.models.nerual_networks.autoencoder import Autoencoder2
from video_embedding.models.video_embedder import VideoEmbedder
from video_embedding.utils import set_session, get_all_names
# import cv2
import PIL
from video_embedding.models.video_embedding_dataset import load_dataloader
# load model src/ILeSiA/video_embedding/saved_models/quantitative_study/peg_place404_model_12.pt
name = "peg_pick404"
set_session("AE3")
video_embedder = VideoEmbedder(
        name=name,
        latent_dim=int(12),
        learning_rate=float(0.001),
        nn_model="Autoencoder3",
    )

video_embedder.load_model()
# video_embedder.load_model(path=Path("src/ILeSiA/video_embedding/saved_models/quantitative_study/peg_door404_Autoencoder2_12.pt"))
    
    
    
# video_embedder.load(videos=["peg_place404"], validation_videos=[], shuffle=False)

# data_iter = iter(video_embedder.dataloader)
# batch0 = data_iter.__next__()

# PIL.Image.fromarray((batch0[0].cpu().numpy()[0].squeeze(0))*255).show()

# make all videos
dataloader = load_dataloader(get_all_names(name), batch_size=64, shuffle=False)
video_embedder.create_video(dataloader)

dataloader = load_dataloader(get_all_names(name)[0], batch_size=64, shuffle=False)

latent_traj = video_embedder.latent_trajectory(dataloader)
video_embedder.visualize_latent_trajectory(latent_traj)




