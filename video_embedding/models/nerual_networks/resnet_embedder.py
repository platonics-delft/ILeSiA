
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

# Remove the last fully connected layer
# Retain all layers except the final fully connected layer
# Decide where to cut the ResNet
class CustomResnetFuns():
    
    def encoder(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
    
    def bencoder(self, x):
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
        torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
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