import torch
import random
import numpy as np
import os
from model import get_model

def set_seed(seed=42):
    # seed 고정
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_checkpoint(path, device, model_type='enhanced', sample_size=256):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # 모델 초기화
    model = get_model(model_type=model_type, sample_size=sample_size).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model