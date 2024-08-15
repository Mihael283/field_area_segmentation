import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_epoch(model, dataset, epoch, device):
    model.eval()
    
    with torch.no_grad():
        # Take a random sample from the dataset
        sample_idx = np.random.randint(len(dataset))
        image, _ = dataset[sample_idx]
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        
        # Get model output
        output = model(image)  # No need to index by 'out' since we're using AdvancedUNet
        
        # Apply sigmoid and threshold to get binary mask
        predicted_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()
        
        # Visualize the input image and the predicted mask
        image = image.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC for visualization
        
        # Rescale image to [0, 1] for visualization
        img_min, img_max = image.min(), image.max()
        image = (image - img_min) / (img_max - img_min + 1e-8)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image)  # No cmap argument for 3-channel image
        ax[0].set_title(f'Epoch {epoch + 1} - Input Image')
        ax[0].axis('off')
        
        ax[1].imshow(predicted_mask, cmap='gray')
        ax[1].set_title(f'Epoch {epoch + 1} - Predicted Mask')
        ax[1].axis('off')
    
    plt.savefig(f'epoch_visualize/epoch_{epoch + 1}_visualization.png')
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
    batch_size = images.size(0)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    
    for i in range(batch_size):
        # Permute the image dimensions to (H, W, 3) for visualization
        img = images[i].permute(1, 2, 0).cpu().numpy()
        
        # Rescale to [0, 1] range
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-8)
        
        axes[i, 0].imshow(img)  # No cmap argument for 3-channel image
        axes[i, 0].set_title(f'Epoch {epoch} - Image {i + 1}')
        
        mask = masks[i].cpu().numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Epoch {epoch} - Mask {i + 1}')
    
    plt.tight_layout()
    plt.savefig(f'training_batches/training_batch_visualization_epoch_{epoch}.png')
    plt.close()
