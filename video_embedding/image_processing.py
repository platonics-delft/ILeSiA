
from torchvision import transforms
import torchvision

# resize images to 64x64
saved_img_processing = transforms.Compose(
    [
        #transforms.ToPILImage(),
        transforms.Resize(
            (64, 64), torchvision.transforms.InterpolationMode.BILINEAR
        ),
        # transforms.Grayscale(num_output_channels=1),
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0),
    ]
)