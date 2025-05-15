import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms.functional import to_pil_image

RUN_BATCHED = True # Set to True if GPU low memory
RUN_BATCHED_SIZE = 16 # Set lower if GPU low memory

class AutoencoderBase(nn.Module):
    def forward(self, x):
        if self.run_batched: return self.bforward(x)
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def bforward(self,x, batch_size=RUN_BATCHED_SIZE):
        ''' Only nneded when GPU low memory '''
        dl = DataLoader(x, batch_size=batch_size)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                latent_images_batch = self.decoder(latent_images_batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)

    def bencoder(self, x, batch_size=RUN_BATCHED_SIZE):
        if not RUN_BATCHED: return self.encoder(x)
        
        dl = DataLoader(x, batch_size=batch_size)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)
    


# Define the autoencoder architecture
class Autoencoder(AutoencoderBase):
    def __init__(self, latent_dim=10, run_batched=RUN_BATCHED):
        self.run_batched = run_batched
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([40, 64, 64]),  # Assuming the input images are 32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([40, 32, 32]),  # After max pooling, the size is halved
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            nn.Flatten(),
            nn.LayerNorm([10240]),  # After flattening, the size is 10240
            nn.ReLU(),
            nn.Linear(int(10240), latent_dim),
            nn.LayerNorm([latent_dim]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, int(10240)),
            nn.LayerNorm([10240]),
            nn.ReLU(),
            nn.Unflatten(1, (40, 16, 16)),
            nn.LayerNorm([40, 16, 16]),
            nn.ConvTranspose2d(40, 40, 
                            kernel_size=3, 
                            stride=2, 
                            padding=1, 
                            output_padding=1),
            nn.LayerNorm([40, 32, 32]),
            nn.ReLU(),
            nn.ConvTranspose2d(40, 1, 
                            kernel_size=3, 
                            stride=2, 
                            padding=1, 
                            output_padding=1),
            nn.LayerNorm([1, 64, 64]),  # Assuming the output images are 64x64
        )

class Autoencoder2(AutoencoderBase):
    def __init__(self, latent_dim: int = 12):
        super(Autoencoder2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [128, 16, 16]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [256, 8, 8]

            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # Output pixel values in [0, 1]
        )


class Autoencoder3(AutoencoderBase):
    def __init__(self, latent_dim: int = 12):
        super(Autoencoder3, self).__init__()  # Fixed class name

        # Encoder with dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Added
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.05),  # Added
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.05),  # Added
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.Dropout(0.02)  # Added after linear layer
        )
        
        # Decoder with dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.Dropout(0.02),  # Added
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.05),  # Added
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Added
            
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

class LargeAutoencoder(AutoencoderBase):
    def __init__(self, latent_dim=10):
        super(LargeAutoencoder, self).__init__()
        # Increasing the number of filters and adding more layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, 64, 64]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([128, 32, 32]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [128, 16, 16]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([256, 16, 16]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [256, 8, 8]

            nn.Flatten(),
            nn.LayerNorm([16384]),  # 256*8*8
            nn.ReLU(),
            nn.Linear(16384, latent_dim),
            nn.LayerNorm([latent_dim]),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16384),
            nn.LayerNorm([16384]),
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LayerNorm([1, 64, 64]),
        )



    

# Example usage
if __name__ == "__main__":
    model = Autoencoder2()
    dummy_input = torch.randn(10, 1, 64, 64)  # Example input (batch_size=1, channels=1, height=64, width=64)
    reconstructed = model(dummy_input)
    print(f"Reconstructed shape: {reconstructed.shape}")