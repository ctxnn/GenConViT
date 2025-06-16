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
    Enhanced version of GenConViT with additional architectural improvements:
    1. Attention mechanism between ED and VAE branches
    2. Feature fusion layer using SwiGLU activation for better integration
    3. LeakyReLU activations to prevent dying ReLU problem
    4. Optional residual connections
    """

    def __init__(self, config, ed, vae, net, fp16, use_attention=True, use_residual=True):
        super(GenConViTV2, self).__init__()
        self.net = net
        self.fp16 = fp16
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Initialize both models for the enhanced architecture
        try:
            self.model_ed = GenConViTED(config)
            self.model_vae = GenConViTVAE(config)
            
            # Load ED weights
            self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
            if 'state_dict' in self.checkpoint_ed:
                self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
            else:
                self.model_ed.load_state_dict(self.checkpoint_ed)
            
            # Load VAE weights
            self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))
            if 'state_dict' in self.checkpoint_vae:
                self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
            else:
                self.model_vae.load_state_dict(self.checkpoint_vae)
            
            # Set models to evaluation mode
            self.model_ed.eval()
            self.model_vae.eval()
            
            if self.fp16:
                self.model_ed.half()
                self.model_vae.half()
                
            # Feature fusion layer - adjust dimensions based on actual output size
            # This will be used to combine features from both models
            feature_dim = config["model"].get("feature_dim", 768)
            
            # Using SwiGLU for feature fusion
            self.fusion = nn.Sequential(
                SwiGLU(feature_dim * 2, feature_dim * 4, feature_dim),
                nn.LayerNorm(feature_dim)
            )
            
            # Cross-attention between ED and VAE outputs
            if self.use_attention:
                self.attention = nn.MultiheadAttention(
                    embed_dim=feature_dim, 
                    num_heads=8, 
                    batch_first=True
                )
                
        except FileNotFoundError as e:
            raise Exception(f"Error: Model weights file not found: {str(e)}")

    def forward(self, x):
        # Get outputs from both models
        x_ed = self.model_ed(x)
        x_vae, _ = self.model_vae(x)
        
        # Apply feature fusion strategies
        if self.net == 'v2':
            if self.use_attention:
                # Reshape tensors for attention if needed
                if len(x_ed.shape) == 2:
                    x_ed_attn = x_ed.unsqueeze(1)  # Add sequence dimension
                    x_vae_attn = x_vae.unsqueeze(1)
                else:
                    x_ed_attn = x_ed
                    x_vae_attn = x_vae
                
                # Apply cross attention between ED and VAE features
                attn_output, _ = self.attention(
                    query=x_ed_attn,
                    key=x_vae_attn,
                    value=x_vae_attn
                )
                
                # If we added a dimension for attention, remove it now
                if len(x_ed.shape) == 2:
                    attn_output = attn_output.squeeze(1)
                
                # Combine original ED features with attention output
                if self.use_residual:
                    combined = torch.cat((x_ed, attn_output), dim=-1)
                else:
                    combined = torch.cat((x_ed, x_vae), dim=-1)
                
                # Apply fusion layer
                output = self.fusion(combined)
                return output
            else:
                # Simple concatenation with fusion layer
                combined = torch.cat((x_ed, x_vae), dim=-1)
                return self.fusion(combined)
        elif self.net == 'ed':
            return x_ed
        elif self.net == 'vae':
            return x_vae
        else:
            # Original behavior for 'genconvit' mode
            return torch.cat((x_ed, x_vae), dim=0)
