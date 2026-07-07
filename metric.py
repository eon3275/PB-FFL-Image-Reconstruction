import torch
import os
import lpips
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
import kagglehub

from dataset import RecursiveImageDataset, get_transforms
from utils import load_checkpoint, set_seed

CONFIG = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'sample_size': 256,
    'batch_size': 32,
    'path': './checkpoints/',
    'model_name' : '256_0.5_32_enhanced.pth',
    'model_type': 'enhanced',
    'data_path': None
}

def main():
    set_seed(42)

    # Dataset
    if CONFIG['data_path'] is None:
        try:
            data_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")
        except:
            data_path = "./data"
    else:
        data_path = CONFIG['data_path']

    transform = get_transforms(CONFIG['sample_size'])
    dataset = RecursiveImageDataset(data_path, transform=transform)
    
    # Test Data
    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size
    _, test_set = random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # Load Model
    print(f"[Info] Loading model: {CONFIG['model_name']}")
    full_path = os.path.join(CONFIG['path'], CONFIG['model_name'])
    model = load_checkpoint(full_path, CONFIG['device'], CONFIG['model_type'], CONFIG['sample_size'])
    
    # Metric
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(CONFIG['device'])
    total_psnr, total_ssim, total_lpips, count = 0, 0, 0, 0
    
    print("[Info] Calculating Metrics...")
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(CONFIG['device'])
            
            # Reconstruction
            posterior = model.encode(images).latent_dist
            recon = model.decode(posterior.mode()).sample
            
            # Normalize to [0, 1]
            imgs_norm = (images * 0.5 + 0.5).clamp(0, 1)
            recon_norm = (recon * 0.5 + 0.5).clamp(0, 1)
            
            # Calculate metrics
            total_psnr += psnr(recon_norm, imgs_norm, data_range=1.0).item() * images.size(0)
            total_ssim += ssim(recon_norm, imgs_norm, data_range=1.0).item() * images.size(0)
            total_lpips += loss_fn_lpips(recon, images).sum().item()
            count += images.size(0)

    # Print Results
    print(f"Model: {CONFIG['model_name']}")
    print(f"PSNR : {total_psnr/count:.4f}")
    print(f"SSIM : {total_ssim/count:.4f}")
    print(f"LPIPS: {total_lpips/count:.4f}")

if __name__ == '__main__':
    main()