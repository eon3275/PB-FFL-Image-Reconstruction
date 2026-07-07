import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torch.utils.data import DataLoader, random_split
import kagglehub

from dataset import RecursiveImageDataset, get_transforms
from utils import load_checkpoint, set_seed

CONFIG = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'sample_size': 256,
    'path' : './checkpoints/',
    'baseline': '256_0.0_32_enhanced.pth',
    'model_type_0': 'enhanced',
    'ours': '256_0.5_32_enhanced.pth', 
    'model_type_1': 'enhanced',
    'crop_coords': (90, 170, 80, 180), # (y1, y2, x1, x2)
    'save_dir': './results' 
}

def visualize_zoom(image, recon_b, recon_p, save_path):
    def to_img(t): return (t.squeeze(0).permute(1, 2, 0).cpu() * 0.5 + 0.5).clamp(0, 1).numpy()
    
    img_gt, img_b, img_p = to_img(image), to_img(recon_b), to_img(recon_p)
    y1, y2, x1, x2 = CONFIG['crop_coords']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles = ["Ground Truth", "Baseline (L1)", "Ours (PB-FFL)"]
    images = [img_gt, img_b, img_p]

    for j in range(3):
        # Original Image
        axes[0, j].imshow(images[j])
        axes[0, j].set_title(titles[j], fontweight='bold', fontsize=14)
        axes[0, j].axis('off')
        
        # Red Box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        axes[0, j].add_patch(rect)

        # Zoomed Crop
        crop = images[j][y1:y2, x1:x2, :]
        axes[1, j].imshow(crop)
        axes[1, j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Zoom comparison saved: {save_path}")
    plt.close()

def visualize_error(image, recon_b, recon_p, save_path):
    VMAX = 0.15   # Error Map 최대 밝기

    def to_tensor_01(t):
        return (t * 0.5 + 0.5).clamp(0, 1)

    gt_t = to_tensor_01(image)
    base_t = to_tensor_01(recon_b)
    prop_t = to_tensor_01(recon_p)

    # Compute Error Map
    err_b = torch.abs(gt_t - base_t).mean(dim=1).squeeze().cpu().numpy()
    err_p = torch.abs(gt_t - prop_t).mean(dim=1).squeeze().cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    cols = ['Ground Truth', 'Baseline (L1)', 'Ours (PB-FFL)']

    # 1. Ground Truth
    axes[0].imshow(gt_t.squeeze().permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(cols[0], fontsize=28, fontweight='bold', pad=20)
    axes[0].axis('off')

    # 2. Baseline Error
    axes[1].imshow(err_b, cmap='inferno', vmin=0, vmax=VMAX)
    axes[1].set_title(cols[1], fontsize=28, fontweight='bold', pad=20)
    axes[1].axis('off')

    # 3. Ours Error
    im_last = axes[2].imshow(err_p, cmap='inferno', vmin=0, vmax=VMAX)
    axes[2].set_title(cols[2], fontsize=28, fontweight='bold', pad=20)
    axes[2].axis('off')

    plt.subplots_adjust(left=0.02, right=0.90, top=0.80, bottom=0.05, wspace=0.05)

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.65])
    cbar = plt.colorbar(im_last, cax=cax)
    cbar.ax.tick_params(labelsize=18)

    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f'Error Map saved: {save_path}')
    plt.close()

def main():
    set_seed(42)
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # Dataset Preparation
    try: data_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")
    except: data_path = "./data"
        
    transform = get_transforms(CONFIG['sample_size'])
    dataset = RecursiveImageDataset(data_path, transform=transform)
    
    # Test Split
    _, test_set = random_split(dataset, [len(dataset)-int(len(dataset)*0.1), int(len(dataset)*0.1)])
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    path_base = os.path.join(CONFIG['path'], CONFIG['baseline'])
    path_ours = os.path.join(CONFIG['path'], CONFIG['ours'])

    model_base = load_checkpoint(path_base, CONFIG['device'], CONFIG['model_type_0'], CONFIG['sample_size'])
    model_ours = load_checkpoint(path_ours, CONFIG['device'], CONFIG['model_type_1'], CONFIG['sample_size'])
    
    images, _ = next(iter(test_loader))
    target_image = images[0].unsqueeze(0).to(CONFIG['device']) 

    with torch.no_grad():
        recon_b = model_base.decode(model_base.encode(target_image).latent_dist.sample()).sample
        recon_p = model_ours.decode(model_ours.encode(target_image).latent_dist.sample()).sample

    # Visualization
    visualize_zoom(target_image, recon_b, recon_p, os.path.join(CONFIG['save_dir'], "Comparison_Zoom.png"))
    visualize_error(target_image, recon_b, recon_p, os.path.join(CONFIG['save_dir'], "Comparison_Error.png"))

if __name__ == '__main__':
    main()