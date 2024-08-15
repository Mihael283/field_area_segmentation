import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from shapely.geometry import Polygon
import rasterio
from torchvision import transforms
import cv2
from skimage.util import random_noise
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from models.advanced_u_net import AdvancedUNet

# Updated TestDataset class to generate 3-channel composite images
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
            blue = src.read(2).astype(float)
            green = src.read(3).astype(float)
            red = src.read(4).astype(float)
            nir = src.read(8).astype(float)
            swir1 = src.read(11).astype(float)
            
            # Calculate NDVI (Normalized Difference Vegetation Index)
            ndvi = (nir - red) / (nir + red + 1e-8)  # Add small epsilon to avoid division by zero
            ndvi = np.nan_to_num(ndvi, nan=0.0)  # Replace NaN with 0.0
            
            # Calculate NDSI (Normalized Difference Snow Index)
            ndsi = (green - swir1) / (green + swir1 + 1e-8)
            ndsi = np.nan_to_num(ndsi, nan=0.0)
            
            # Urban False Color (using NIR, Red, Green)
            urban_false_color = np.dstack((nir, red, green))
            urban_false_color = (urban_false_color - urban_false_color.min()) / (urban_false_color.max() - urban_false_color.min() + 1e-8)
            urban_false_color = np.nan_to_num(urban_false_color, nan=0.0)
            
            # Create a 3-channel composite image (NDVI, NDSI, Red from Urban False Color)
            composite = np.dstack((ndvi, ndsi, urban_false_color[:, :, 1]))
            composite = (composite - composite.min()) / (composite.max() - composite.min() + 1e-8)
            composite = np.nan_to_num(composite, nan=0.0)
        
        # Apply transform if available
        if self.transform:
            composite = self.transform(composite)
        
        return composite, file_name

def find_polygons(mask):
    # Ensure the mask is binary
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Find contours of white areas
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.0001 * cv2.arcLength(contour, True)  # Much smaller epsilon for more detail
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygons.append(approx)
    
    return polygons

def polygon_to_segmentation(polygon):
    """Convert an OpenCV polygon to a segmentation list."""
    return polygon.flatten().tolist()

def predict_and_save(model, test_loader, device, output_file):
    model.eval()
    results = {"images": []}

    with torch.no_grad():
        for composite, file_names in tqdm(test_loader, desc="Processing test images"):
            composite = composite.to(device).float()
            
            batch_size, channels, height, width = composite.shape
            
            pad_h = (32 - height % 32) % 32
            pad_w = (32 - width % 32) % 32
            composite_padded = torch.nn.functional.pad(composite, (0, pad_w, 0, pad_h), mode='reflect')
            
            outputs = model(composite_padded)
            output = outputs[:, :, :height, :width]  # Remove padding
            
            predicted = (torch.sigmoid(output) > 0.5).float()

            for i in range(batch_size):
                pred_mask = predicted[i, 0].cpu().numpy()
                polygons = find_polygons(pred_mask)

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
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])
test_dataset = TestDataset(test_image_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedUNet(n_channels=3, n_classes=1).to(device) 
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Run predictions and save results
predict_and_save(model, test_loader, device, 'results.json')

print("Predictions completed. Results saved in 'results.json'")