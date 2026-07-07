import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class PB_FFL(nn.Module):
    def __init__(self, weight=1.0, alpha=1.0, patch_size=32, stride=16):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.patch_size = patch_size
        # Sliding Window
        self.unfolder = nn.Unfold(kernel_size=(patch_size, patch_size), stride=stride)
        # Hann Window
        hann = torch.hann_window(patch_size, periodic=True)
        window_2d = hann.unsqueeze(1) * hann.unsqueeze(0) # broadcasting
        self.register_buffer("window", window_2d.view(1, 1, patch_size, patch_size))

    def forward(self, input, target):
        #L1 Loss
        loss_spatial = F.l1_loss(input, target)
        if self.weight <= 0: # Use Baseline
            return loss_spatial
        input_f = input.float()
        target_f = target.float()
        B, C, H, W = input_f.shape
        # Patch Extraction , Shape=[Batch, Channel * Patch_H * Patch_W, Num_Patches]
        input_patches = self.unfolder(input_f)
        target_patches = self.unfolder(target_f)
        # Reshape: [B, C, Patch_H, Patch_W, Num_Patches] -> [Batch*Num_Patches, C, Patch_H, Patch_W]
        input_patches = input_patches.view(B, C, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3).reshape(-1, C, self.patch_size, self.patch_size)
        target_patches = target_patches.view(B, C, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3).reshape(-1, C, self.patch_size, self.patch_size)

        # FFT with Hann Windowing
        input_fft = torch.fft.fft2(input_patches * self.window)
        target_fft = torch.fft.fft2(target_patches * self.window)

        # Calculate frequency distance
        dist = torch.abs(input_fft - target_fft)
        
        # Apply Dynamic Weighting
        weight = torch.log((dist ** self.alpha) + 1.0)
        weight = weight / (weight.max() + 1e-8) # Normalize

        # Get Frequency Loss
        loss_freq = (weight * (dist ** 2)).mean()

        return loss_spatial + self.weight * loss_freq