import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from timm import create_model
from model.config import load_config
from .model_embedder import HybridEmbed

config = load_config()

class SquaredLeakyReLU(nn.Module):
    """
    Squared LeakyReLU activation function.
    Computes: (LeakyReLU(x))²
    """
    def __init__(self, negative_slope=0.01, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        
    def forward(self, x):
        # Apply LeakyReLU then square the result
        x = F.leaky_relu(x, self.negative_slope, self.inplace)
        return x * x

class EncoderV2(nn.Module):
    """
    Modified VAE Encoder with squared LeakyReLU
    """
    def __init__(self, latent_dims=4):
        super(EncoderV2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            SquaredLeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            SquaredLeakyReLU(negative_slope=0.01),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            SquaredLeakyReLU(negative_slope=0.01),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            SquaredLeakyReLU(negative_slope=0.01)
        )

        self.latent_dims = latent_dims
        self.fc1 = nn.Linear(128*14*14, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128*14*14, self.latent_dims)
        self.var = nn.Linear(128*14*14, self.latent_dims)

        self.kl = 0
        self.kl_weight = 0.5  # 0.00025
        self.activation = SquaredLeakyReLU(negative_slope=0.01)

    def reparameterize(self, x):
        # https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py
        std = torch.exp(0.5*self.mu(x))
        eps = torch.randn_like(std)
        z = eps * std + self.mu(x)

        return z, std
        
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        
        z, std = self.reparameterize(x)
        
        return z

class DecoderV2(nn.Module):
    """
    Modified VAE Decoder with squared LeakyReLU
    """
    def __init__(self, latent_dims=4):
        super(DecoderV2, self).__init__()

        self.latent_dims = latent_dims
        
        self.features = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dims, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            SquaredLeakyReLU(negative_slope=0.01),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            SquaredLeakyReLU(negative_slope=0.01),
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            SquaredLeakyReLU(negative_slope=0.01),
            
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            SquaredLeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        x = x.view(-1, self.latent_dims, 1, 1)
        
        return self.features(x)

class GenConViTVAEV2(nn.Module):
    """
    Modified GenConViTVAE with squared LeakyReLU activation
    """
    def __init__(self, config):
        super(GenConViTVAEV2, self).__init__()
        
        model_name = config["model"]["backbone"]
        embed_dim = config["model"]["feature_dim"]
        stride = config["model"]["stride"]
        self.encoder = EncoderV2(latent_dims=embed_dim)
        self.decoder = DecoderV2(latent_dims=embed_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        return z, x_hat
