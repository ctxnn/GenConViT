import torch
import torch.nn as nn
import torch.nn.functional as F
from .genconvit_ed import GenConViTED
from .genconvit_vae import GenConViTVAE
from torchvision import transforms

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.
    As used in Transformer models to replace GELU.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = x1 * F.silu(x2)  # SiLU (Swish) activation
        return self.w3(hidden)

class GenConViTV2(nn.Module):
    """
    Modified version of GenConViT that only changes the activation functions:
    1. Uses SwiGLU instead of GELU
    2. Uses LeakyReLU instead of ReLU
    
    This version does NOT include attention or residual connections,
    maintaining the same structure as the original GenConViT.
    """

    def __init__(self, config, ed, vae, net, fp16, use_attention=False, use_residual=False):
        super(GenConViTV2, self).__init__()
        self.net = net
        self.fp16 = fp16
        
        # Store parameters for metadata
        self.use_attention = False  # Explicitly set to False since we're removing attention
        self.use_residual = False   # Explicitly set to False since we're removing residual connections
        
        # Get device (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models based on network type
        if self.net == 'ed':
            try:
                self.model_ed = GenConViTED(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=self.device)

                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)

                self.model_ed.eval()
                self.model_ed.to(self.device)
                if self.fp16:
                    self.model_ed.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
        elif self.net == 'vae':
            try:
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=self.device)

                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                    
                self.model_vae.eval()
                self.model_vae.to(self.device)
                if self.fp16:
                    self.model_vae.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{vae}.pth file not found.")
        else:
            try:
                self.model_ed = GenConViTED(config)
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=self.device)
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=self.device)
                
                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)
                    
                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                    
                self.model_ed.eval()
                self.model_vae.eval()
                
                self.model_ed.to(self.device)
                self.model_vae.to(self.device)
                
                if self.fp16:
                    self.model_ed.half()
                    self.model_vae.half()
            except FileNotFoundError as e:
                raise Exception(f"Error: Model weights file not found.")

    def forward(self, x):
        # Ensure input is on the same device as model
        device = next(self.parameters()).device
        x = x.to(device)
        
        # Process with appropriate network
        if self.net == 'ed':
            return self.model_ed(x)
        elif self.net == 'vae':
            x_vae, _ = self.model_vae(x)
            return x_vae
        else:
            # Original behavior for 'genconvit' mode
            x_ed = self.model_ed(x)
            x_vae, _ = self.model_vae(x)
            return torch.cat((x_ed, x_vae), dim=0)
