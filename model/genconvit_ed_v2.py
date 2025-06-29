import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
import timm
from .model_embedder import HybridEmbed
import torch.nn.functional as F

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
    Modified Encoder using SquaredLeakyReLU instead of ReLU
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(    
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class DecoderV2(nn.Module):
    """
    Modified Decoder using SquaredLeakyReLU instead of ReLU
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            SquaredLeakyReLU(negative_slope=0.01, inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class GenConViTEDV2(nn.Module):
    """
    Modified GenConViTED with SquaredLeakyReLU instead of standard activations
    """
    def __init__(self, config):
        super(GenConViTEDV2, self).__init__()
        
        model_name = config["model"]["backbone"]
        embed_dim = config["model"]["feature_dim"]
        stride = config["model"]["stride"]
        self.encoder = EncoderV2()
        self.decoder = DecoderV2()
        self.fc = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, embed_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use SwiGLU instead of GELU
        self.activation = nn.SiLU()

    def forward(self, x):
        batch_size = x.shape[0]
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        x = self.encoder(x)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = self.fc2(self.activation(self.fc(self.activation(x))))
        return x
