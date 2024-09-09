
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self,image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and os.path.exists(os.path.join(image_dir, f.replace('.jpg', '_seg.png'))) ]    
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.image_dir, self.images[idx].replace('.jpg', '_seg.png'))

        image = np.array(Image.open(img_path).convert("RGB")) # for using albumentations use np.array
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask ==255.0] = 1.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask= mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
