from video_embedding.utils import set_session
from video_embedding.models.video_embedder import VideoEmbedder
from risk_estimation.models.safety_layer import SafetyLayer
import torch
import matplotlib.pyplot as plt
import numpy as np

from risk_estimation.models.risk_estimation.saliency_map_generator import get_saliency_map_for_image

def plot_saliency(image: np.ndarray, 
                  saliency: np.ndarray):
    
    assert image.shape == saliency.shape

    # Create a plot to display the image and the saliency map
    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the grayscale image
    ax.imshow(image, cmap='gray', interpolation='nearest')

    # Overlay the saliency map
    # 'alpha' controls the transparency of the overlay
    ax.imshow(saliency, cmap='hot', alpha=0.5, interpolation='nearest')  # cmap='hot' for the saliency

    # Remove axes and display the figure
    ax.axis('off')
    plt.show()



if __name__ == '__main__':
    set_session("my_new_session")
    safety_model = SafetyLayer()
    model = safety_model.video_embedder.model
    
    # Image loading and preprocessing
    image = safety_model.get_random_image()

    saliency = get_saliency_map_for_image(model, image)

    plot_saliency(image.squeeze().detach().cpu().numpy(),
                  saliency.cpu().numpy(),
    )
