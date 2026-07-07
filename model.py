import torch
from diffusers import AutoencoderKL

def get_model(model_type='enhanced', sample_size=256):
    # Returns AutoencoderKL
    if model_type == 'enhanced':
        # Enhanced Model
        channels = [128, 256, 512]
    else:
        # Base Model
        channels = [64, 128, 256]

    # Model Initialization
    model = AutoencoderKL(
        in_channels=3, 
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"]*3, 
        up_block_types=["UpDecoderBlock2D"]*3,
        block_out_channels=channels, 
        layers_per_block=1, 
        latent_channels=4,
        act_fn="silu", 
        norm_num_groups=32, 
        sample_size=sample_size
    )
    
    return model