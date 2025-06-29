import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from model.config import load_config
from model.model_embedder import HybridEmbed

config = load_config()

class Encoder(nn.Module):
    def __init__(self, latent_dims=4, img_size=224):
        super(Encoder, self).__init__()
        
        self.img_size = img_size
        self.latent_dims = latent_dims
        
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(inplace=True)
        )
        
        # Calculate the flattened size dynamically
        self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc_intermediate = nn.Linear(self.flattened_size, 512)
        self.mu = nn.Linear(512, self.latent_dims)
        self.logvar = nn.Linear(512, self.latent_dims)
        
        # KL divergence parameters
        self.kl = 0
        self.kl_weight = 0.0001
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def _calculate_flattened_size(self):
        """Calculate the flattened size after convolutional layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.img_size, self.img_size)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Convolutional layers
        x = self.features(x)
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
    def __init__(self, latent_dims=4, img_size=224):
        super(Decoder, self).__init__()
        
        self.img_size = img_size
        self.latent_dims = latent_dims
        
        # Calculate the size needed for upsampling
        self.init_size = img_size // 16  # Size after downsampling
        self.decoder_input_size = 128 * self.init_size * self.init_size
        
        # Initial linear layer
        self.decoder_input = nn.Linear(latent_dims, self.decoder_input_size)
        
        # Reshape layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, self.init_size, self.init_size))
        
        # Transpose convolutional layers
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
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
        
        # Configuration
        self.img_size = config.get('img_size', 224)
        self.latent_dims = config['model']['latent_dims']
        self.num_classes = config.get('num_classes', 2)
        
        # VAE components
        self.encoder = Encoder(self.latent_dims, self.img_size)
        self.decoder = Decoder(self.latent_dims, self.img_size)
        
        # Backbone network
        try:
            self.embedder = create_model(config['model']['embedder'], pretrained=pretrained)
            self.convnext_backbone = create_model(
                config['model']['backbone'], 
                pretrained=pretrained, 
                num_classes=1000, 
                drop_path_rate=0, 
                head_init_scale=1.0
            )
            self.convnext_backbone.patch_embed = HybridEmbed(
                self.embedder, 
                img_size=self.img_size, 
                embed_dim=768
            )
        except Exception as e:
            print(f"Warning: Could not initialize backbone models: {e}")
            # Fallback to simple CNN backbone
            self.use_simple_backbone = True
            self.simple_backbone = self._create_simple_backbone()
        else:
            self.use_simple_backbone = False
        
        # Get feature dimensions
        if not self.use_simple_backbone:
            try:
                if hasattr(self.convnext_backbone, 'head') and hasattr(self.convnext_backbone.head, 'fc'):
                    backbone_features = self.convnext_backbone.head.fc.out_features
                elif hasattr(self.convnext_backbone, 'head'):
                    backbone_features = self.convnext_backbone.head.in_features
                else:
                    backbone_features = 1000
            except:
                backbone_features = 1000
        else:
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
        
        # Transform for resizing
        self.resize_transform = transforms.Resize((224, 224), antialias=True)
        
        # Loss weights
        self.recon_weight = 1.0
        self.kl_weight = 0.0001
        self.cls_weight = 1.0

    def _create_simple_backbone(self):
        """Create a simple CNN backbone as fallback"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512)
        )

    def forward(self, x):
        # Encode
        z, mu, logvar = self.encoder(x)
        
        # Decode
        x_reconstructed = self.decoder(z)
        
        # Resize for backbone processing
        x_resized = self.resize_transform(x)
        x_recon_resized = self.resize_transform(x_reconstructed)
        
        # Feature extraction
        if not self.use_simple_backbone:
            try:
                features_original = self.convnext_backbone(x_resized)
                features_reconstructed = self.convnext_backbone(x_recon_resized)
            except Exception as e:
                print(f"Warning: Backbone forward failed: {e}, using simple backbone")
                self.use_simple_backbone = True
                features_original = self.simple_backbone(x_resized)
                features_reconstructed = self.simple_backbone(x_recon_resized)
        else:
            features_original = self.simple_backbone(x_resized)
            features_reconstructed = self.simple_backbone(x_recon_resized)
        
        # Concatenate features
        combined_features = torch.cat([features_original, features_reconstructed], dim=1)
        
        # Classification
        classification_output = self.classifier(combined_features)
        
        return classification_output, x_reconstructed, mu, logvar

    def get_loss(self, x, labels, classification_output, x_reconstructed, mu, logvar):
        """Calculate combined loss"""
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