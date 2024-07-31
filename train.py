import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from models.u_net import UNet
from dataset import SatelliteDataset
from helper import visualize_epoch, plot_losses
from logger import Logger

# Set up CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logger
logger = Logger(log_dir="logs")

# Set up the dataset and data loaders
image_dir = 'train_images/images'
annotation_file = 'train_annotation.json'

transform = transforms.Compose([
    transforms.Normalize(mean=[0.485], std=[0.229])
])

full_dataset = SatelliteDataset(image_dir, annotation_file, transform=transform)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

batch_size = 6
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the model, loss function, and optimizer
model = UNet(n_channels=1, n_classes=1).to(device)  # Single channel output for binary segmentation
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Log configurations
logger.log_model_config(model)
logger.log_optimizer_config(optimizer)
logger.log_scheduler_config(scheduler)
logger.log_dataset_config(full_dataset, train_size, val_size, batch_size)
logger.log_training_config(num_epochs=10, criterion=criterion, device=device)

# Training loop
num_epochs = 10
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
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
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), 'best_model.pth')
    
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
    
    # Log epoch results
    logger.log_epoch(epoch+1, train_loss, val_loss, current_lr)
    
    # Visualize epoch results
    visualize_epoch(model, full_dataset, epoch, device)

# Plot and save losses
plot_losses(train_losses, val_losses, num_epochs)

# Log final results
logger.log_final_results(best_epoch+1, best_val_loss)

print(f"Best model saved at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
print("Training complete. Best model saved as 'best_model.pth' and loss plot saved as 'loss_plot.png'")
print(f"Training log saved at {logger.log_file}")