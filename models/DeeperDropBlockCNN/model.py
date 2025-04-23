import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    """
    Implements DropBlock: a form of structured dropout for convolutional layers
    that drops entire regions of feature maps instead of individual pixels.
    """
    def __init__(self, block_size, drop_prob):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob
        
    def forward(self, x):
        # If we're not in training mode or drop_prob is 0, just return input
        if not self.training or self.drop_prob == 0:
            return x
            
        # Get dimensions
        batch_size, channels, height, width = x.size()
        
        # Calculate keep probability (probability of keeping a region)
        gamma = self.drop_prob * (height * width) / (self.block_size ** 2) / \
                ((height - self.block_size + 1) * (width - self.block_size + 1))
        
        # Sample mask
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Place mask on device
        mask = mask.to(x.device)
        
        # Create block mask by dilating the sampled mask
        block_mask = F.max_pool2d(mask, kernel_size=self.block_size, 
                                 stride=1, padding=self.block_size//2)
        
        # Invert the mask to get regions to keep (1 means keep, 0 means drop)
        block_mask = 1 - block_mask
        
        # Scale output to preserve activations during inference
        norm_factor = block_mask.sum() / (batch_size * channels * height * width)
        if norm_factor > 0:
            block_mask = block_mask / norm_factor
        
        # Apply the mask to input tensor
        return x * block_mask
        
class DeeperDropBlockCNN(nn.Module):
    def __init__(self, num_classes, drop_prob=0.1, block_size=7):
        super(DeeperDropBlockCNN, self).__init__()
        self.out_features = 256
        
        # Create DropBlock layer
        self.dropblock = DropBlock2D(block_size=block_size, drop_prob=drop_prob)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, self.out_features, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output shape: [batch, 256, 4, 4]
        
        self.fc1 = nn.Linear(self.out_features * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.ndim == 3:  # [batch, 256, 256] → add channel dimension
            x = x.unsqueeze(1)
            
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropblock(x)  # Apply DropBlock after first block
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropblock(x)  # Apply DropBlock after second block
        
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropblock(x)  # Apply DropBlock after third block
        
        x = self.pool(self.relu(self.conv4(x)))
        # No DropBlock after final conv layer, before classification
        
        x = self.adaptive_pool(x)  # [B, 256, 4, 4]
        x = x.view(x.size(0), -1)  # Flatten → [B, 4096]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x