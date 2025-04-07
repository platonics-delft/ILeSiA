
from video_embedding.models.nerual_networks.autoencoder import Autoencoder
import torch

def get_saliency_map_for_image(model: Autoencoder,
                               image: torch.tensor, # (1,64,64)
                               ):  

    # Forward pass
    image = image.unsqueeze(0) # shape (1,1,64,64)
    image.requires_grad_(True)
    
    output = model(image)

    # Zero the parameter gradients
    model.zero_grad()

    # Backward pass
    output_idx_flatten = output.argmax()

    output_idx_x, output_idx_y = output_idx_flatten // 64, output_idx_flatten % 64 
    output[0, 0, output_idx_x, output_idx_y].backward()

    # Saliency map
    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    saliency = saliency.reshape(64, 64)

    return saliency
