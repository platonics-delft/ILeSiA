import torch
import torchvision.transforms as transforms

# Define a transformation pipeline for batched tensors
resize_transform = transforms.Compose(
    [
        transforms.Resize((64, 64), transforms.InterpolationMode.BILINEAR),  # Resize all images in the batch
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
    ]
)

# Example batch of images (B, C, H, W)
batch_images = torch.randn(16, 3, 128, 128)  # Batch of 16 images with 3 channels

# Apply the transformation directly to the batch
transformed_batch = resize_transform(batch_images)

print(transformed_batch.shape)  # Should be (16, 1, 64, 64)