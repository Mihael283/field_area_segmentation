import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from shapely.geometry import Polygon
import skimage.draw

class SatelliteDataset(Dataset):
    def __init__(self, image_paths, annotations):
        self.image_paths = image_paths
        self.annotations = annotations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            with rasterio.open(image_path) as src:
                image = src.read()
                rgb_bands = [3, 2, 1]  # Assuming B4=Red, B3=Green, B2=Blue
                rgb_image = np.dstack([image[i-1] for i in rgb_bands])  # rasterio uses 1-based indexing
                rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)
                image_tensor = torch.from_numpy(rgb_image).float().permute(2, 0, 1) / 255.0
        except RasterioIOError:
            print(f"Error reading image: {image_path}")
            return None, None
        
        boxes = []
        masks = []
        for ann in self.annotations[idx]:
            try:
                segmentation = ann['segmentation']
                if len(segmentation) < 6 or len(segmentation) % 2 != 0:  # At least 3 points needed
                    print(f"Skipping invalid segmentation in image {image_path}")
                    continue
                
                coords = np.array(segmentation).reshape(-1, 2)
                polygon = Polygon(coords)
                
                if not polygon.is_valid or polygon.is_empty:
                    print(f"Skipping invalid polygon in image {image_path}")
                    continue
                
                mask = self.polygon_to_mask(polygon, image_tensor.shape[1:])
                if np.sum(mask) == 0:
                    print(f"Skipping empty mask in image {image_path}")
                    continue
                
                masks.append(mask)
                
                bounds = polygon.bounds
                boxes.append([bounds[0], bounds[1], bounds[2], bounds[3]])
            except Exception as e:
                print(f"Error processing annotation in image {image_path}: {e}")
                continue
        
        if not boxes:
            print(f"No valid annotations found for image {image_path}")
            return None, None
        
        # Convert list of masks to a single numpy array
        masks = np.array(masks, dtype=np.uint8)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.from_numpy(masks)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all are 'field' class
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        
        return image_tensor, target

    def polygon_to_mask(self, polygon, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        coords = np.array(polygon.exterior.coords).astype(int)
        
        # Clip coordinates to image boundaries
        coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[1] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[0] - 1)
        
        rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0])
        
        # Clip rr and cc to image boundaries
        rr = np.clip(rr, 0, image_shape[0] - 1)
        cc = np.clip(cc, 0, image_shape[1] - 1)
        
        mask[rr, cc] = 1
        return mask

def collate_fn(batch):
    return tuple(zip(*[b for b in batch if b[0] is not None]))