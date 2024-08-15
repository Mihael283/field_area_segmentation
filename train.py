import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.advanced_u_net import AdvancedUNet
from dataset import SatelliteDataset
from helper import visualize_epoch, plot_losses, visualize_training_batch
from logger import Logger
import numpy as np
import torch.nn.functional as F

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = Logger(log_dir="logs")
num_epochs = 50

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

batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # Ensure pred and target have the same shape
        if pred.dim() == 4 and target.dim() == 3:
            target = target.unsqueeze(1)
        
        bce = self.bce_loss(pred, target)
        pred = torch.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        return self.alpha * bce + (1 - self.alpha) * dice

    def dice_loss(self, pred, target, smooth=1.):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))
        return dice

# Usage
model = AdvancedUNet(n_channels=3, n_classes=1).to(device) 

criterion = CombinedLoss(alpha=0.5)  # You can adjust alpha as needed
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Log configurations
logger.log_model_config(model)
logger.log_optimizer_config(optimizer)
logger.log_scheduler_config(scheduler)
logger.log_dataset_config(full_dataset, train_size, val_size, batch_size)
logger.log_training_config(num_epochs=num_epochs, criterion="Combined BCE and Dice Loss", device=device)

def compute_epoch_dice(model, dataloader, device):
    model.eval()
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for ndvi, mask in dataloader:
            ndvi, mask = ndvi.to(device), mask.to(device)
            outputs = model(ndvi)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use only the main output for evaluation
            pred = torch.sigmoid(outputs)
            dice = 1 - criterion.dice_loss(pred, mask.unsqueeze(1))  # Add channel dimension to mask
            total_dice += dice.item()
            num_batches += 1
    
    return total_dice / num_batches


train_losses = []
val_losses = []
val_dice_scores = []
best_val_dice = 0.0


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    first_batch = next(iter(train_loader))
    visualize_training_batch(first_batch[0], first_batch[1], epoch+1)
    
    for ndvi, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        ndvi, mask = ndvi.to(device), mask.to(device)
        
        optimizer.zero_grad()
        outputs, deep2, deep3, deep4 = model(ndvi)
        loss = criterion(outputs, mask)
        loss += 0.5 * criterion(F.interpolate(deep2, size=mask.shape[1:]), mask)
        loss += 0.3 * criterion(F.interpolate(deep3, size=mask.shape[1:]), mask)
        loss += 0.2 * criterion(F.interpolate(deep4, size=mask.shape[1:]), mask)
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
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use only the main output for validation
            loss = criterion(outputs, mask)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    val_dice = compute_epoch_dice(model, val_loader, device)
    val_dice_scores.append(val_dice)
    
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with validation Dice score: {best_val_dice:.4f}")
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, LR: {current_lr:.6f}")
    
    logger.log_epoch(epoch+1, train_loss, val_loss, val_dice, current_lr)
    logger.save_log()
    
    visualize_epoch(model, full_dataset, epoch, device)

plot_losses(train_losses, val_losses, val_dice_scores, num_epochs)

logger.log_final_results(epoch+1, best_val_dice)
logger.save_log() 

print(f"Best model saved with validation Dice score: {best_val_dice:.4f}")
print("Training complete. Best model saved as 'best_model.pth' and plots saved as 'training_plots.png'")
print(f"Training log saved at {logger.log_file}")