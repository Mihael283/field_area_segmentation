import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon
import skimage.draw
import os
import json
from sklearn.model_selection import train_test_split
from pq_calculate import compute_pq
from dataset import SatelliteDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_image_folder = 'train_images/images'
annotation_file = 'train_annotations.json'

image_paths = [os.path.join(train_image_folder, f) for f in os.listdir(train_image_folder) if f.endswith('.tif')]

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

image_to_annotation = {item['file_name']: item['annotations'] for item in annotations['images']}

dataset = [(path, image_to_annotation[os.path.basename(path)]) for path in image_paths]

train_data, val_data = train_test_split(dataset, test_size=0.3, random_state=42)

train_dataset = SatelliteDataset([item[0] for item in train_data], [item[1] for item in train_data])
val_dataset = SatelliteDataset([item[0] for item in val_data], [item[1] for item in val_data])

train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# Model setup
num_classes = 2  # Background and field
model = maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    
    for images, targets in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        pred_masks = model(images)[0]['masks']
        gt_masks = targets[0]['masks']

        pred_polygons = []
        gt_polygons = []

        for mask in pred_masks:
            mask_np = mask.squeeze().cpu().numpy()
            contours = skimage.measure.find_contours(mask_np, 0.5)
            if contours:
                polygon = Polygon(contours[0])
                if polygon.is_valid and polygon.area > 0:
                    pred_polygons.append(polygon)

        for mask in gt_masks:
            mask_np = mask.squeeze().cpu().numpy()
            contours = skimage.measure.find_contours(mask_np, 0.5)
            if contours:
                polygon = Polygon(contours[0])
                if polygon.is_valid and polygon.area > 0:
                    gt_polygons.append(polygon)

        pq, _, _ = compute_pq(gt_polygons, pred_polygons)
        pq_loss = 1 - pq

        losses += pq_loss
        total_train_loss += losses.item()
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    avg_train_loss = total_train_loss / len(train_data_loader)
    
    # Validation loop
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(val_data_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
    
    avg_val_loss = total_val_loss / len(val_data_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("Training complete!")