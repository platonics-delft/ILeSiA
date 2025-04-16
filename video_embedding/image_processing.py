
from torchvision import transforms
import torchvision
import torch

# resize images to 64x64
def saved_img_processing(img):
    min_dim_size = min(img.shape[0], img.shape[1])
    resize_transform = transforms.Compose(
        [
            transforms.CenterCrop((min_dim_size, min_dim_size)),
            transforms.Resize(
                (64, 64), torchvision.transforms.InterpolationMode.BILINEAR
            ),
        ]
    )

    img_tensor = torch.tensor(img, dtype=torch.float32).cuda().unsqueeze(0)
    return resize_transform(img_tensor) / 255.0