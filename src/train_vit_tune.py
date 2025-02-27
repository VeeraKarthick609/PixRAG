import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np
import optuna

from torch.cuda.amp import autocast, GradScaler
from models.vit import ViT

# -------------------------
# Mixup Data Augmentation Functions
# -------------------------
def mixup_data(x, y, alpha):
    """Return mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -------------------------
# Training and Validation Functions
# -------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, scheduler, mixup_alpha, clip_grad_norm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        scheduler.step()  # Step scheduler each iteration
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -------------------------
# Optuna Objective Function for Hyperparameter Tuning
# -------------------------
def objective(trial):
    # Sample hyperparameters
    max_lr = trial.suggest_loguniform("max_lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    mixup_alpha = trial.suggest_uniform("mixup_alpha", 0.0, 0.4)
    label_smoothing = trial.suggest_uniform("label_smoothing", 0.0, 0.2)
    clip_grad_norm = trial.suggest_uniform("clip_grad_norm", 0.5, 2.0)
    
    num_epochs = 20  # Use fewer epochs for tuning
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data transforms (standard CIFAR-100 normalization)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])
    
    train_dataset = datasets.CIFAR100(root="data/cifar100", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="data/cifar100", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=100,
                embed_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps,
                                              pct_start=0.1, anneal_strategy='cos', final_div_factor=100)
    
    scaler = GradScaler()
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                                device, scaler, scheduler, mixup_alpha, clip_grad_norm)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val_acc = max(best_val_acc, val_acc)
    return best_val_acc

# -------------------------
# Main: Run Hyperparameter Tuning
# -------------------------
def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Adjust number of trials as needed
    print("Best trial:")
    trial = study.best_trial
    print("  Best Validation Accuracy: {:.4f}".format(trial.value))
    print("  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # You can now use these best hyperparameters for a full training run.
    # For example, save them to a JSON file for later use.
    best_params_path = "checkpoints/best_hyperparameters.txt"
    os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
    with open(best_params_path, "w") as f:
        f.write("Best Hyperparameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
    print(f"Best hyperparameters saved to {best_params_path}")

if __name__ == "__main__":
    main()
