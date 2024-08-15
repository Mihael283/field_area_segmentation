import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import cv2
import json
import random
from scipy.ndimage import rotate
from skimage.util import random_noise

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
        h, w = image.shape[:2]  # Get the height and width of the image
        target_h, target_w = self.target_size  # The target size is (800, 800)
        
        scale = min(target_h / h, target_w / w)  # Compute the scaling factor
        
        new_h, new_w = int(h * scale), int(w * scale)  # Compute the new height and width after scaling
        resized = cv2.resize(image, (new_w, new_h))  # Resize the image
        
        pad_h = (target_h - new_h) // 2  # Compute padding for height
        pad_w = (target_w - new_w) // 2  # Compute padding for width
        
        # Apply padding to each channel. Note the use of np.pad with ((...)) to pad all three dimensions.
        padded = np.pad(resized, ((pad_h, target_h - new_h - pad_h), 
                                (pad_w, target_w - new_w - pad_w), 
                                (0, 0)), 
                        mode='constant', constant_values=0)
        
        return padded, (pad_h, pad_w, new_h, new_w)

    def apply_augmentation(self, composite, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            composite = np.fliplr(composite)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            composite = np.flipud(composite)
            mask = np.flipud(mask)
        
        # Random rotation
        rotation = random.choice([0, 20, 45, 75, 90, 120, 180, 210, 270])
        composite = rotate(composite, rotation, reshape=False)
        mask = rotate(mask, rotation, reshape=False)
        
        # Apply random noise
        noise_type = random.choice(['gaussian', 'salt', 'pepper', 's&p', 'localvar', 'speckle', 'poisson'])
        composite = random_noise(composite, mode=noise_type)
        
        # Ensure that the composite is still in the [0, 1] range after noise application
        composite = np.clip(composite, 0, 1)
        
        return np.ascontiguousarray(composite), np.ascontiguousarray(mask)
    
    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = f"{self.image_dir}/{img_info['file_name']}"
        
        with rasterio.open(img_path) as src:
            blue = src.read(2).astype(float)
            green = src.read(3).astype(float)
            red = src.read(4).astype(float)
            nir = src.read(8).astype(float)
            swir1 = src.read(11).astype(float)
            
            # Calculate NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
            ndvi = np.nan_to_num(ndvi, nan=0.0)  # Replace NaN with 0.0
            
            # Calculate NDSI (Normalized Difference Snow Index)
            ndsi = (green - swir1) / (green + swir1 + 1e-8)  # Add small epsilon to avoid division by zero
            ndsi = np.nan_to_num(ndsi, nan=0.0)  # Replace NaN with 0.0
            
            # Urban False Color (using NIR, Red, Green)
            urban_false_color = np.dstack((nir, red, green))
            urban_false_color = (urban_false_color - urban_false_color.min()) / (urban_false_color.max() - urban_false_color.min() + 1e-8)
            urban_false_color = np.nan_to_num(urban_false_color, nan=0.0)  # Replace NaN with 0.0
            
            # Create a 3-channel composite image (NDVI, NDSI, Red from Urban False Color)
            composite = np.dstack((ndvi, ndsi, urban_false_color[:, :, 1]))
            composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
            composite = np.nan_to_num(composite, nan=0.0)  # Replace NaN with 0.0

        composite_padded, (pad_h, pad_w, new_h, new_w) = self.resize_and_pad(composite)
        
        mask = np.zeros(self.target_size, dtype=np.uint8)

        for ann in img_info['annotations']:
            coords = np.array(ann['segmentation']).reshape(-1, 2)
            
            coords[:, 0] = coords[:, 0] * new_w / composite.shape[1] + pad_w
            coords[:, 1] = coords[:, 1] * new_h / composite.shape[0] + pad_h
            
            coords = coords.astype(int)
            cv2.fillPoly(mask, [coords], 1)
        
        if self.augment:
            composite_padded, mask = self.apply_augmentation(composite_padded, mask)
        
        composite_tensor = torch.from_numpy(composite_padded).float().permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).float()

        # Final check for NaN or Inf and replace them if found
        composite_tensor = torch.nan_to_num(composite_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        mask_tensor = torch.nan_to_num(mask_tensor, nan=0.0, posinf=1.0, neginf=0.0)
        
        assert not torch.isnan(composite_tensor).any(), f"NaN detected in the input image at index {idx}"
        assert not torch.isinf(composite_tensor).any(), f"Inf detected in the input image at index {idx}"
        assert not torch.isnan(mask_tensor).any(), f"NaN detected in the mask at index {idx}"
        assert not torch.isinf(mask_tensor).any(), f"Inf detected in the mask at index {idx}"

        if self.transform:
            composite_tensor = self.transform(composite_tensor)
        
        return composite_tensor, mask_tensor