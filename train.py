import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.u_net import UNet
from dataset import SatelliteDataset
from helper import visualize_epoch, plot_losses, visualize_training_batch
from logger import Logger
from pq_calculate import compute_pq
import numpy as np
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = Logger(log_dir="logs")
num_epochs = 20

image_dir = 'train_images/images'
annotation_file = 'train_annotation.json'

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485], std=[0.229])
])


full_dataset = SatelliteDataset(image_dir, annotation_file, transform=transform, augment=False)

train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])

train_dataset = SatelliteDataset(image_dir, annotation_file, transform=transform, augment=True)
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

batch_size = 6
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

model = UNet(n_channels=1, n_classes=1).to(device) 
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Log configurations
logger.log_model_config(model)
logger.log_optimizer_config(optimizer)
logger.log_scheduler_config(scheduler)
logger.log_dataset_config(full_dataset, train_size, val_size, batch_size)
logger.log_training_config(num_epochs=num_epochs, criterion=criterion, device=device)

import cv2
from shapely.geometry import Polygon

def getIOU(polygon1: Polygon, polygon2: Polygon) -> float:
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0
    try:
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        if union == 0:
            return 0
        return intersection / union
    except Exception:
        return 0

def mask_to_polygons(mask, min_area=10):
    """
    Convert a binary mask to a list of polygons.
    
    Args:
    mask (numpy.ndarray): A 2D binary numpy array where 1 indicates the object.
    min_area (int): Minimum area (in pixels) for a polygon to be included.
    
    Returns:
    list: A list of Shapely Polygon objects.
    """
    mask = (mask > 0.5).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) < 3: 
            continue
        poly = Polygon(contour.squeeze())
        
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type != 'Polygon':
                continue
        
        if poly.area > min_area:
            poly = poly.simplify(1.0, preserve_topology=True)
            polygons.append(poly)
    
    return polygons


def compute_epoch_pq(model, dataloader, device):
    model.eval()
    all_gt_polygons = []
    all_pred_polygons = []
    
    with torch.no_grad():
        for ndvi, mask in dataloader:
            ndvi, mask = ndvi.to(device), mask.to(device)
            outputs = model(ndvi)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            
            for i in range(mask.shape[0]):
                gt_polygons = mask_to_polygons(mask[i].cpu().numpy())
                pred_polygons = mask_to_polygons(predicted[i].cpu().numpy())
                all_gt_polygons.extend(gt_polygons)
                all_pred_polygons.extend(pred_polygons)
    
    try:
        pq, sq, rq = compute_pq(all_gt_polygons, all_pred_polygons)
    except Exception as e:
        print(f"Error computing PQ score: {e}")
        pq, sq, rq = 0, 0, 0
    
    return pq, sq, rq

train_losses = []
val_losses = []
val_pq_scores = []
best_val_pq = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    first_batch = next(iter(train_loader))
    visualize_training_batch(first_batch[0], first_batch[1], epoch+1)
    
    for ndvi, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        ndvi, mask = ndvi.to(device), mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(ndvi)
        loss = criterion(outputs, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ndvi, mask in val_loader:
            ndvi, mask = ndvi.to(device), mask.to(device)
            outputs = model(ndvi)
            loss = criterion(outputs, mask)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    val_pq, val_sq, val_rq = compute_epoch_pq(model, val_loader, device)
    val_pq_scores.append(val_pq)
    
    if val_pq > best_val_pq:
        best_val_pq = val_pq
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with validation PQ: {best_val_pq:.4f}")
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PQ: {val_pq:.4f}, LR: {current_lr:.6f}")
    
    logger.log_epoch(epoch+1, train_loss, val_loss, val_pq, val_sq, val_rq, current_lr)
    logger.save_log()
    
    visualize_epoch(model, full_dataset, epoch, device)


plot_losses(train_losses, val_losses, val_pq_scores, num_epochs)

logger.log_final_results(epoch+1, best_val_pq)
logger.save_log() 

print(f"Best model saved with validation PQ: {best_val_pq:.4f}")
print("Training complete. Best model saved as 'best_model.pth' and plots saved as 'training_plots.png'")
print(f"Training log saved at {logger.log_file}")