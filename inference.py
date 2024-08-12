import torch
import json
import os
from tqdm import tqdm
from models.advanced_u_net import AdvancedUNet
from torch.utils.data import Dataset, DataLoader
import numpy as np
from shapely.geometry import Polygon
import rasterio
from torchvision import transforms
from PIL import Image
import cv2

class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.file_names = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.image_dir, file_name)
        
        with rasterio.open(img_path) as src:
            red = src.read(4).astype(float)
            nir = src.read(8).astype(float)
            ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Convert NDVI to 0-255 range
        ndvi = (ndvi * 255).astype(np.uint8)
        
        if self.transform:
            ndvi = self.transform(ndvi)
        
        print(f"NDVI shape in dataset: {ndvi.shape}")  # Debug print
        
        return ndvi, file_name

def mask_to_polygons(mask, min_area=10):
    mask = (mask > 0.5).astype(np.uint8) * 255  # Ensure mask is 0 or 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) < 3:  # A polygon needs at least 3 points
            continue
        poly = Polygon(contour.squeeze())
        
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.geom_type != 'Polygon':
                continue
        
        if poly.area > min_area:
            poly = poly.simplify(1.0, preserve_topology=True)
            polygons.append(poly)
    
    return polygons

def polygon_to_segmentation(polygon):
    """Convert a Shapely polygon to a segmentation list."""
    coords = np.array(polygon.exterior.coords)
    return coords.flatten().tolist()

def predict_and_save(model, test_loader, device, output_file):
    model.eval()
    results = {"images": []}

    with torch.no_grad():
        for ndvi, file_names in tqdm(test_loader, desc="Processing test images"):
            ndvi = ndvi.to(device)
            
            # Print shape for debugging
            print(f"Input shape: {ndvi.shape}")
            
            # Get current image size
            if ndvi.dim() == 4:  # (B, C, H, W)
                batch_size, channels, height, width = ndvi.shape
            elif ndvi.dim() == 3:  # (C, H, W)
                channels, height, width = ndvi.shape
                batch_size = 1
                ndvi = ndvi.unsqueeze(0)
            else:
                raise ValueError(f"Unexpected input shape: {ndvi.shape}")
            
            # Pad the input to make it divisible by 32 (assuming your U-Net has 5 down/up-sampling operations)
            pad_h = (32 - height % 32) % 32
            pad_w = (32 - width % 32) % 32
            ndvi_padded = torch.nn.functional.pad(ndvi, (0, pad_w, 0, pad_h), mode='reflect')
            
            outputs = model(ndvi_padded)
            
            # Print shape for debugging
            print(f"Output shape: {outputs.shape}")
            
            # Remove padding from the output
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take only the main output, not the deep supervision outputs
            outputs = outputs[:, :, :height, :width]
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(batch_size):
                pred_mask = predicted[i, 0].cpu().numpy()  # Take the first channel
                polygons = mask_to_polygons(pred_mask)

                image_result = {
                    "file_name": file_names[i],
                    "annotations": []
                }

                for polygon in polygons:
                    annotation = {
                        "class": "field",
                        "segmentation": polygon_to_segmentation(polygon)
                    }
                    image_result["annotations"].append(annotation)

                results["images"].append(image_result)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Set up the test dataset and data loader
test_image_dir = 'test_images/images'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
test_dataset = TestDataset(test_image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedUNet(n_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

# Run predictions and save results
predict_and_save(model, test_loader, device, 'results.json')

print("Predictions completed. Results saved in 'results.json'")