import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class RecursiveImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        self.transform = transform
        
    def __len__(self): 
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            if self.transform: 
                img = self.transform(img)
            return img, 0 
        except Exception as e:
            return self.__getitem__(0) 

def get_transforms(size=256):
    return transforms.Compose([
        # transforms.Resize((size, size)) #256x256 Dataset이므로 생략
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])