import torch
import torchvision
from torch.utils.data import  Dataset
import numpy as np
import skimage.draw

class SatelliteDataset(Dataset):
    def __init__(self, image_paths, annotations):
        self.image_paths = image_paths
        self.annotations = annotations
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = torchvision.io.read_image(self.image_paths[idx]).float() / 255.0
        
        boxes = []
        masks = []
        for ann in self.annotations[idx]:
            mask = np.zeros(image.shape[1:], dtype=np.uint8)
            for polygon in ann['segmentation']:
                coords = np.array(polygon).reshape(-1, 2)
                rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0])
                mask[rr, cc] = 1
            masks.append(mask)
            
            pos = np.where(mask)
            xmin, xmax = np.min(pos[1]), np.max(pos[1])
            ymin, ymax = np.min(pos[0]), np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.uint8)
        labels = torch.ones((len(boxes),), dtype=torch.int64)  # Assuming all are 'field' class
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        
        return image, target