import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_epoch(model, dataset, epoch, device):
    model.eval()
    ndvi, true_mask = dataset[0]  # Get the first sample for visualization
    ndvi = ndvi.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(ndvi)
        predicted_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(ndvi.squeeze().cpu().numpy(), cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('NDVI')
    ax1.axis('off')
    
    ax2.imshow(true_mask.squeeze().numpy(), cmap='binary')
    ax2.set_title('True Mask')
    ax2.axis('off')
    
    ax3.imshow(predicted_mask, cmap='binary')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')
    
    plt.savefig(f'epoch_{epoch+1}_visualization.png')
    plt.close()

def plot_losses(train_losses, val_losses, num_epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()