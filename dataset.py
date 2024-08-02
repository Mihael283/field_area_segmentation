import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import cv2
import json
import random
from scipy.ndimage import rotate

class SatelliteDataset(Dataset):
    def __init__(self, image_dir, annotation_file, target_size=(800, 800), transform=None, augment=False):
        self.image_dir = image_dir
        self.transform = transform
        self.target_size = target_size
        self.augment = augment
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)['images']
        
        print(f"Dataset initialized with {len(self.annotations)} samples, augment={augment}")
    
    def __len__(self):
        return len(self.annotations)
    
    def resize_and_pad(self, image):
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        scale = min(target_h / h, target_w / w)
        
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        padded = np.pad(resized, ((pad_h, target_h - new_h - pad_h), 
                                  (pad_w, target_w - new_w - pad_w)), 
                        mode='constant', constant_values=0)
        
        return padded, (pad_h, pad_w, new_h, new_w)
    
    def apply_augmentation(self, ndvi, mask):
        if random.random() > 0.5:
            ndvi = np.fliplr(ndvi)
            mask = np.fliplr(mask)
        if random.random() > 0.5:
            ndvi = np.flipud(ndvi)
            mask = np.flipud(mask)
        
        rotation = random.choice([0, 10, 20, 30,45])
        ndvi = rotate(ndvi, rotation, reshape=False)
        mask = rotate(mask, rotation, reshape=False)            
        
        return np.ascontiguousarray(ndvi), np.ascontiguousarray(mask)
    
    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = f"{self.image_dir}/{img_info['file_name']}"
        
        with rasterio.open(img_path) as src:
            red = src.read(4).astype(float)
            nir = src.read(8).astype(float)
            ndvi = (nir - red) / (nir + red)
            ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-8)
            ndvi = np.nan_to_num(ndvi, nan=0.0)
        
        ndvi_padded, (pad_h, pad_w, new_h, new_w) = self.resize_and_pad(ndvi)
        
        mask = np.zeros(self.target_size, dtype=np.uint8)

        for ann in img_info['annotations']:
            coords = np.array(ann['segmentation']).reshape(-1, 2)
            
            coords[:, 0] = coords[:, 0] * new_w / ndvi.shape[1] + pad_w
            coords[:, 1] = coords[:, 1] * new_h / ndvi.shape[0] + pad_h
            
            coords = coords.astype(int)
            cv2.fillPoly(mask, [coords], 1)
        
        if self.augment:
            ndvi_padded, mask = self.apply_augmentation(ndvi_padded, mask)
        
        ndvi_tensor = torch.from_numpy(ndvi_padded).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float()
        
        if self.transform:
            ndvi_tensor = self.transform(ndvi_tensor)
        
        return ndvi_tensor, mask_tensor