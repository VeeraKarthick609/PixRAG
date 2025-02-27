import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np

from torch.cuda.amp import autocast, GradScaler
from models.vit import ViT

# -------------------------
# Mixup Data Augmentation Functions
# -------------------------
def mixup_data(x, y, alpha):
    """Returns mixed inputs, pairs of targets, and lambda."""
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

        scheduler.step()  # Scheduler steps on every batch

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
# Main Training Loop (using best hyperparameters from tuning)
# -------------------------
def main():
    # Best hyperparameters obtained from tuning (example values)
    num_epochs = 100
    batch_size = 128
    max_lr = 0.0007          # Best maximum learning rate
    weight_decay = 3e-4        # Best weight decay
    mixup_alpha = 0.3          # Best mixup alpha
    label_smoothing = 0.1      # Best label smoothing value
    clip_grad_norm = 1.0       # Gradient clipping norm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data augmentation: Using RandAugment if available; otherwise, standard augmentations.
    try:
        from torchvision.transforms import RandAugment
        rand_augment = RandAugment(num_ops=2, magnitude=9)
        print("Using RandAugment")
    except ImportError:
        rand_augment = lambda x: x  # No-op if RandAugment isn't available
        print("RandAugment not available; skipping")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        rand_augment,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

    # Load CIFAR-100 datasets
    train_dataset = datasets.CIFAR100(root="data/cifar100", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root="data/cifar100", train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize the custom ViT model
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=100,
                embed_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1)
    model.to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    
    # OneCycleLR Scheduler: total_steps is total batches * num_epochs
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=100
    )
    
    scaler = GradScaler()
    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            scheduler, mixup_alpha, clip_grad_norm
        )
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/vit_cifar100_best.pth")
            print("Saved best model")
    
    print("Training complete. Best validation accuracy: {:.4f}".format(best_acc))

if __name__ == "__main__":
    main()
