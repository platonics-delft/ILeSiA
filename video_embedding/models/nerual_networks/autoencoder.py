import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms.functional import to_pil_image


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=10):
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
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_batched(self,x):
        ''' Only nneded when GPU low memory '''
        dl = DataLoader(x, batch_size = 1)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                latent_images_batch = self.decoder(latent_images_batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)

    def encoder_batched(self, x):
        ''' Only nneded when GPU low memory '''
        dl = DataLoader(x, batch_size = 1)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)



class LargeAutoencoder(nn.Module):
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
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_batched(self,x):
        ''' Only nneded when GPU low memory '''
        dl = DataLoader(x, batch_size = 1)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                latent_images_batch = self.decoder(latent_images_batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)

    def encoder_batched(self, x):
        dl = DataLoader(x, batch_size = 1)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)

# Remove the last fully connected layer
# Retain all layers except the final fully connected layer
# Decide where to cut the ResNet


class CustomResnetFuns():
    
    def encoder(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
    
    def encoder_batched(self, x):
        dl = DataLoader(x, batch_size = 1)
        out = []
        with torch.no_grad():
            for batch in dl:
                latent_images_batch = self.encoder(batch)
                out.append(latent_images_batch)
        return torch.cat(out, dim=0)
    
    def to_3channel(self, x):
        out = []
        for x_ in x:
            x_pil = to_pil_image(x_)
            x__ = self.rgb_transform(x_pil)
            x__ = torch.tensor(256 * (1-x__), dtype=torch.float32).cuda()
            out.append(x__.unsqueeze(0))
        return torch.cat(out)

    rgb_transform = resnet_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


class CustomResnetStage1(CustomResnetFuns, nn.Module):
    def __init__(self, latent_dim):
        super(CustomResnetStage1, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:4])  # Ends at stage 1 64
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.to_3channel(x)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x


class CustomResnetStage2(CustomResnetFuns, nn.Module):
    def __init__(self, latent_dim):
        super(CustomResnetStage2, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:5])  # Ends at stage 2 256
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.to_3channel(x)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x

class CustomResnetStage3(CustomResnetFuns, nn.Module):
    def __init__(self, latent_dim):
        super(CustomResnetStage3, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:6])  # Ends at stage 3 512
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.to_3channel(x)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x
    
class CustomResnetStage4(CustomResnetFuns, nn.Module):
    def __init__(self, latent_dim):
        super(CustomResnetStage4, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:7])  # Ends at stage 4 1024
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.to_3channel(x)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x

class CustomResnetStage5(CustomResnetFuns, nn.Module):
    def __init__(self, latent_dim):
        super(CustomResnetStage5, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # All stages 2048
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.to_3channel(x)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x