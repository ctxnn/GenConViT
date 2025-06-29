import torch
import torch.nn as nn
from genconvit_ed_final import GenConViTED
from genconvit_vae_final import GenConViTVAE
from torchvision import transforms

class GenConViT(nn.Module):

    def __init__(self, config, ed, vae, net, fp16):
        super(GenConViT, self).__init__()
        self.net = net
        self.fp16 = fp16
        self.config = config
        
        if self.net == 'ed':
            try:
                self.model_ed = GenConViTED(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))

                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)

                self.model_ed.eval()
                if self.fp16:
                    self.model_ed.half()
                    
                print(f"Successfully loaded ED model from weight/{ed}.pth")
                
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
            except Exception as e:
                raise Exception(f"Error loading ED model: {str(e)}")
                
        elif self.net == 'vae':
            try:
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))

                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                    
                self.model_vae.eval()
                if self.fp16:
                    self.model_vae.half()
                    
                print(f"Successfully loaded VAE model from weight/{vae}.pth")
                
            except FileNotFoundError:
                raise Exception(f"Error: weight/{vae}.pth file not found.")
            except Exception as e:
                raise Exception(f"Error loading VAE model: {str(e)}")
                
        else:  # Load both models for ensemble
            try:
                self.model_ed = GenConViTED(config)
                self.model_vae = GenConViTVAE(config)
                
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))
                
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
                
                if self.fp16:
                    self.model_ed.half()
                    self.model_vae.half()
                    
                print(f"Successfully loaded both ED and VAE models")
                
            except FileNotFoundError as e:
                raise Exception(f"Error: Model weights file not found: {str(e)}")
            except Exception as e:
                raise Exception(f"Error loading models: {str(e)}")

    def forward(self, x):
        """
        Forward pass through the model(s)
        """
        # Ensure input is properly sized (128x128 for our models)
        if x.size(2) != 128 or x.size(3) != 128:
            x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        
        if self.net == 'ed':
            # ED model returns (classification_output, reconstructed)
            classification_output, _ = self.model_ed(x)
            return classification_output
            
        elif self.net == 'vae':
            # VAE model returns (classification_output, x_reconstructed, mu, logvar)
            classification_output, _, _, _ = self.model_vae(x)
            return classification_output
            
        else:
            # Ensemble: combine both models
            ed_output, _ = self.model_ed(x)
            vae_output, _, _, _ = self.model_vae(x)
            
            # Average the predictions
            combined_output = (ed_output + vae_output) / 2.0
            return combined_output

    def get_model_info(self):
        """
        Get information about the loaded model(s)
        """
        info = {
            'net_type': self.net,
            'fp16': self.fp16,
            'input_size': '128x128',
        }
        
        if hasattr(self, 'model_ed'):
            ed_params = sum(p.numel() for p in self.model_ed.parameters())
            info['ed_parameters'] = f"{ed_params:,}"
            
        if hasattr(self, 'model_vae'):
            vae_params = sum(p.numel() for p in self.model_vae.parameters())
            info['vae_parameters'] = f"{vae_params:,}"
            
        return info