import torch
import torch.nn as nn
import numpy as np
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

class ViTWithCellposeHead(nn.Module):
    def __init__(self, base_vit_model, additional_branch_output_dim):
        super(ViTWithCellposeHead, self).__init__()
        self.base_vit_model = base_vit_model
        # Assuming 'decoder_channels[-1]' is the dimension of the final feature maps from ViT_seg
        in_channels = base_vit_model.config.decoder_channels[-1]
        self.additional_branch = nn.Sequential(
            # Example architecture for the additional branch
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, additional_branch_output_dim, kernel_size=1)
        )
        
    def forward(self, x):
        # Forward pass through the base ViT model
        segmentation_output, features = self.base_vit_model(x)
        
        # Forward pass through the additional branch
        additional_output = self.additional_branch(features)
        
        return segmentation_output, additional_output

# Example of setting up the model
config_vit = get_r50_b16_config()  # Assuming this is your method to get the config
net = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes)  # Initialize your ViT model
net.load_from(weights=np.load(config_vit.pretrained_path))

# Wrap your ViT model with the additional branch
model_with_additional_head = ViTWithCellposeHead(net, additional_branch_output_dim=3)  # '2' as an example dimension
model_with_additional_head.cuda()  # Move the model to GPU
