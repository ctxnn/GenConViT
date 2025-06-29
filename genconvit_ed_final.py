import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import timm
from model.config import load_config

config = load_config()

class Encoder(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()
        
        self.img_size = img_size
        
        # Encoder designed for 128x128 input
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

    def forward(self, x):
        return self.features(x)

class Decoder(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()
        
        self.img_size = img_size
        
        # Decoder to reconstruct 128x128 images
        self.features = nn.Sequential(
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

    def forward(self, x):
        return self.features(x)

class GenConViTED(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTED, self).__init__()
        
        # Configuration
        self.img_size = 128  # Fixed to match your data
        self.num_classes = config.get('num_classes', 2)
        
        # Encoder-Decoder components
        self.encoder = Encoder(self.img_size)
        self.decoder = Decoder(self.img_size)
        
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
        self.num_features = backbone_features * 2
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self.num_features // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.num_features // 2, self.num_features // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.num_features // 4, self.num_classes)
        )
        
        # Reconstruction loss weight
        self.recon_weight = 1.0
        self.cls_weight = 1.0

    def forward(self, images):
        # Ensure input is 128x128 (your data size)
        if images.size(2) != self.img_size or images.size(3) != self.img_size:
            # Resize to match expected input size
            images = nn.functional.interpolate(images, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Encode-Decode
        encoded = self.encoder(images)
        decoded = self.decoder(encoded)
        
        # Feature extraction from original and reconstructed images
        features_original = self.backbone(images)
        features_reconstructed = self.backbone(decoded)
        
        # Concatenate features
        combined_features = torch.cat([features_original, features_reconstructed], dim=1)
        
        # Classification
        classification_output = self.classifier(combined_features)
        
        return classification_output, decoded
    
    def get_loss(self, images, labels, classification_output, reconstructed):
        """Calculate combined loss for ED model"""
        # Ensure both tensors have the same size for reconstruction loss
        if images.size() != reconstructed.size():
            reconstructed = nn.functional.interpolate(reconstructed, size=(images.size(2), images.size(3)), mode='bilinear', align_corners=False)
        
        # Reconstruction loss (MSE)
        recon_loss = nn.MSELoss()(reconstructed, images)
        
        # Classification loss
        cls_loss = nn.CrossEntropyLoss()(classification_output, labels)
        
        # Combined loss
        total_loss = self.recon_weight * recon_loss + self.cls_weight * cls_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'cls_loss': cls_loss
        }