import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from model.config import load_config
from model.model_embedder import HybridEmbed

config = load_config()

class Encoder(nn.Module):
    def __init__(self, latent_dims=4, img_size=128):
        super(Encoder, self).__init__()
        
        self.img_size = img_size
        self.latent_dims = latent_dims
        
        # Convolutional layers designed for 128x128 input
        self.features = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Fixed flattened size: 512 * 4 * 4 = 8192
        self.flattened_size = 512 * 4 * 4
        
        # Fully connected layers
        self.fc_intermediate = nn.Linear(self.flattened_size, 1024)
        self.mu = nn.Linear(1024, self.latent_dims)
        self.logvar = nn.Linear(1024, self.latent_dims)
        
        # KL divergence parameters
        self.kl = 0
        self.kl_weight = 0.0001
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.2)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Convolutional layers
        x = self.features(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Intermediate layer
        x = self.relu(self.fc_intermediate(x))
        x = self.dropout(x)
        
        # Get mu and logvar
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        # Calculate KL divergence
        self.kl = self.kl_weight * torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims=4, img_size=128):
        super(Decoder, self).__init__()
        
        self.img_size = img_size
        self.latent_dims = latent_dims
        
        # Start from 4x4 feature maps
        self.init_size = 4
        self.decoder_input_size = 512 * self.init_size * self.init_size
        
        # Initial linear layer
        self.decoder_input = nn.Linear(latent_dims, self.decoder_input_size)
        
        # Reshape layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, self.init_size, self.init_size))
        
        # Transpose convolutional layers to reconstruct 128x128 images
        self.decoder_conv = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, z):
        x = self.decoder_input(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

class GenConViTVAE(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTVAE, self).__init__()
        
        # Configuration - use actual input size from data
        self.img_size = 128  # Fixed to match your data
        self.latent_dims = config['model']['latent_dims']
        self.num_classes = config.get('num_classes', 2)
        
        # VAE components
        self.encoder = Encoder(self.latent_dims, self.img_size)
        self.decoder = Decoder(self.latent_dims, self.img_size)
        
        # Simple backbone for feature extraction from 128x128 images
        self.backbone = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Feature dimensions
        backbone_features = 512
        self.num_feature = backbone_features * 2
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_feature, self.num_feature // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.num_feature // 2, self.num_feature // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.num_feature // 4, self.num_classes)
        )
        
        # Loss weights
        self.recon_weight = 1.0
        self.kl_weight = 0.0001
        self.cls_weight = 1.0

    def forward(self, x):
        # Ensure input is 128x128 (your data size)
        if x.size(2) != self.img_size or x.size(3) != self.img_size:
            # Resize to match expected input size
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Encode
        z, mu, logvar = self.encoder(x)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        # Feature extraction from original and reconstructed images
        # Both should be 128x128 now
        features_original = self.backbone(x)
        features_reconstructed = self.backbone(x_reconstructed)
        
        # Concatenate features
        combined_features = torch.cat([features_original, features_reconstructed], dim=1)
        
        # Classification
        classification_output = self.classifier(combined_features)
        
        return classification_output, x_reconstructed, mu, logvar

    def get_loss(self, x, labels, classification_output, x_reconstructed, mu, logvar):
        """Calculate combined loss"""
        # Ensure both tensors have the same size for reconstruction loss
        if x.size() != x_reconstructed.size():
            x_reconstructed = nn.functional.interpolate(x_reconstructed, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(x_reconstructed, x)
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)  # Normalize by batch size
        
        # Classification loss
        cls_loss = nn.CrossEntropyLoss()(classification_output, labels)
        
        # Combined loss
        total_loss = (self.recon_weight * recon_loss + 
                     self.kl_weight * kl_loss + 
                     self.cls_weight * cls_loss)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'cls_loss': cls_loss
        }