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
    
    plt.savefig(f'epoch_visualize/epoch_{epoch+1}_visualization.png')
    plt.close()

def plot_losses(train_losses, val_losses, val_pq_scores, num_epochs):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs+1), val_pq_scores, label='Validation PQ')
    plt.xlabel('Epoch')
    plt.ylabel('PQ Score')
    plt.title('Validation PQ Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_plots.png')
    plt.close()


def visualize_training_batch(images, masks, epoch):
    """
    Visualize a batch of training images and their corresponding masks.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        if i < images.shape[0]:
            # NDVI Image
            axes[0, i].imshow(images[i].squeeze().cpu().numpy(), cmap='viridis')
            axes[0, i].set_title(f"NDVI {i+1}")
            axes[0, i].axis('off')
            
            # Mask
            axes[1, i].imshow(masks[i].squeeze().cpu().numpy(), cmap='gray')
            axes[1, i].set_title(f"Mask {i+1}")
            axes[1, i].axis('off')
        else:
            axes[0, i].axis('off')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'training_batches/training_batch_visualization_epoch_{epoch}.png')
    plt.close()
