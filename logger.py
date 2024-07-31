import json
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.log_data = {
            "training_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": [],
            "model_config": {},
            "optimizer_config": {},
            "scheduler_config": {},
            "dataset_config": {},
            "training_config": {}
        }

    def log_model_config(self, model):
        self.log_data["model_config"] = {
            "name": model.__class__.__name__,
            "n_channels": model.n_channels,
            "n_classes": model.n_classes
        }

    def log_optimizer_config(self, optimizer):
        self.log_data["optimizer_config"] = {
            "name": optimizer.__class__.__name__,
            "lr": optimizer.param_groups[0]["lr"],
            "betas": optimizer.param_groups[0].get("betas", None),
            "weight_decay": optimizer.param_groups[0].get("weight_decay", None)
        }

    def log_scheduler_config(self, scheduler):
        self.log_data["scheduler_config"] = {
            "name": scheduler.__class__.__name__,
            "factor": scheduler.factor,
            "patience": scheduler.patience
        }

    def log_dataset_config(self, dataset, train_size, val_size, batch_size):
        self.log_data["dataset_config"] = {
            "image_dir": dataset.image_dir,
            "train_size": train_size,
            "val_size": val_size,
            "batch_size": batch_size
        }

    def log_training_config(self, num_epochs, criterion, device):
        self.log_data["training_config"] = {
            "num_epochs": num_epochs,
            "criterion": criterion.__class__.__name__,
            "device": str(device)
        }

    def log_epoch(self, epoch, train_loss, val_loss, lr):
        self.log_data["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr
        })

    def save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def log_final_results(self, best_epoch, best_val_loss):
        self.log_data["final_results"] = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "training_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_log()