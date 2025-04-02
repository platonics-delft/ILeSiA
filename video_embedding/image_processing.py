
from torchvision import transforms
import torchvision
import torch

# resize images to 64x64
def saved_img_processing(img):
    
    resize_transform = transforms.Compose(
        [
            transforms.Resize(
                (64, 64), torchvision.transforms.InterpolationMode.BILINEAR
            ),
        ]
    )

    img_tensor = torch.tensor(img, dtype=torch.float32).cuda().unsqueeze(0)
    return resize_transform(img_tensor) / 255.0