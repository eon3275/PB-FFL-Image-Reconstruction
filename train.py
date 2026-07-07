import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import kagglehub

from dataset import RecursiveImageDataset, get_transforms
from model import get_model
from loss import PB_FFL
from utils import set_seed

CONFIG = {
    'device': "cuda" if torch.cuda.is_available() else "cpu",
    'sample_size': 256,
    'batch_size': 64,
    'lr': 1e-4,
    'epochs': 20,
    'patch_size': 32,
    'stride': 16,
    'alpha': 1.0,
    'freq_loss_weight': 0.5,
    'model_type': 'enhanced',
    'save_dir': './checkpoints'
}

def main():
    set_seed(42)
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    try:
        dataset_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")
    except Exception as e:
        print(f"Dataset Download Failed: {e}")
        return

    transform = get_transforms(CONFIG['sample_size'])
    full_dataset = RecursiveImageDataset(dataset_path, transform=transform)
    
    # Train/Val Split (9:1)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )

    print(f"Device: {CONFIG['device']}")
    
    # Model & Loss Initialization
    model = get_model(model_type=CONFIG['model_type'], sample_size=CONFIG['sample_size']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    objective_function = PB_FFL(
        weight=CONFIG['freq_loss_weight'], 
        alpha=CONFIG['alpha'], 
        patch_size=CONFIG['patch_size'], 
        stride=CONFIG['stride']
    ).to(CONFIG['device'])
    
    scaler = torch.amp.GradScaler('cuda') 

    # Training Loop
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for images, _ in loop:
            images = images.to(CONFIG['device'], non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                # Forward
                posterior = model.encode(images).latent_dist
                reconstruction = model.decode(posterior.sample()).sample
                
                # Loss = L1 + PB_FFL + KL
                loss = objective_function(reconstruction, images) + 1e-6 * posterior.kl().mean()
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            if loop.n % 100 == 0:
                loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Save Checkpoint
    save_name = f"{CONFIG['sample_size']}_{CONFIG['freq_loss_weight']}_{CONFIG['patch_size']}_{CONFIG['model_type']}.pth"
    save_path = os.path.join(CONFIG['save_dir'], save_name)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': CONFIG
    }
    torch.save(checkpoint, save_path)
    print(f"[Info] Model saved to {save_path}")

if __name__ == '__main__':
    main()
