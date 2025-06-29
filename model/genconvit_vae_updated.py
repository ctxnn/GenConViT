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

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU()
        )

        self.latent_dims = latent_dims
        self.img_size = img_size
        
        # Calculate the size after convolutions dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu = nn.Linear(self.flattened_size, self.latent_dims)
        self.var = nn.Linear(self.flattened_size, self.latent_dims)

        self.kl = 0
        self.kl_weight = 0.5
        self.relu = nn.LeakyReLU()

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.mu(x)
        var = self.var(x)
        z = self.reparameterize(mu, var)
        
        self.kl = self.kl_weight * torch.mean(-0.5 * torch.sum(1 + var - mu**2 - var.exp(), dim=1), dim=0) 
        
        return z

class Decoder(nn.Module):
  
    def __init__(self, latent_dims=4, img_size=224):
        super(Decoder, self).__init__()

        self.latent_dims = latent_dims
        self.img_size = img_size
        
        # Calculate the size needed for reconstruction
        final_conv_size = img_size // 16  # After 4 stride-2 convolutions
        self.final_conv_size = final_conv_size
        self.decoder_input_size = 128 * final_conv_size * final_conv_size
        
        self.decoder_input = nn.Linear(latent_dims, self.decoder_input_size)
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128, final_conv_size, final_conv_size))

        self.features = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Tanh()  # Use Tanh for better reconstruction
        )

    def forward(self, x): 
        x = self.decoder_input(x)
        x = self.unflatten(x)
        x = self.features(x)
        return x
        
class GenConViTVAE(nn.Module):
    def __init__(self, config, pretrained=True):
        super(GenConViTVAE, self).__init__()
        
        # Get configuration parameters
        self.img_size = config['img_size']
        self.latent_dims = config['model']['latent_dims']
        
        # Initialize encoder and decoder with image size
        self.encoder = Encoder(self.latent_dims, self.img_size)
        self.decoder = Decoder(self.latent_dims, self.img_size)
        
        # Initialize the embedder and backbone
        self.embedder = create_model(config['model']['embedder'], pretrained=True)
        self.convnext_backbone = create_model(config['model']['backbone'], pretrained=True, num_classes=1000, drop_path_rate=0, head_init_scale=1.0)
        self.convnext_backbone.patch_embed = HybridEmbed(self.embedder, img_size=config['img_size'], embed_dim=768)
        
        # Get the number of features from the backbone
        if hasattr(self.convnext_backbone, 'head') and hasattr(self.convnext_backbone.head, 'fc'):
            backbone_features = self.convnext_backbone.head.fc.out_features
        elif hasattr(self.convnext_backbone, 'head'):
            backbone_features = self.convnext_backbone.head.in_features
        else:
            backbone_features = 1000  # Default fallback
            
        self.num_feature = backbone_features * 2
 
        # Classification layers
        self.fc = nn.Linear(self.num_feature, self.num_feature//4)
        self.fc3 = nn.Linear(self.num_feature//2, self.num_feature//4)
        self.fc2 = nn.Linear(self.num_feature//4, config['num_classes'])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Resize transform for backbone input
        self.resize = transforms.Resize((224, 224), antialias=True)

    def forward(self, x):
        # VAE forward pass
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        # Resize images for backbone processing
        x_resized = self.resize(x)
        x_hat_resized = self.resize(x_hat)

        # Forward pass through backbone
        x1 = self.convnext_backbone(x_resized)
        x2 = self.convnext_backbone(x_hat_resized)
        
        # Concatenate features
        x = torch.cat((x1, x2), dim=1)
        
        # Classification layers
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, self.resize(x_hat)