import os
import json
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# CONFIG
# =========================
@dataclass
class CFG:
    # Dataset root
    base_dir: str = r"C:\datasets\real_fake_faces\archive\real_vs_fake\real-vs-fake"

    img_size: int = 128
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4

    # Output
    run_dir: str = "runs/simplecnn_run1"


cfg = CFG()


# =========================
# UTILS
# =========================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj, path: str):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# =========================
# MODEL
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, img_size=128, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.bad = 0
            return True
        self.bad += 1
        if self.bad >= self.patience:
            self.should_stop = True
        return False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
    return np.array(preds), np.array(labels)


# =========================
# MAIN
# =========================
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Paths
    train_dir = os.path.join(cfg.base_dir, "train")
    val_dir = os.path.join(cfg.base_dir, "valid")
    test_dir = os.path.join(cfg.base_dir, "test")

    # Sanity check
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Valid dir not found: {val_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test dir not found: {test_dir}")

    # Output
    ensure_dir(cfg.run_dir)
    ckpt_best = os.path.join(cfg.run_dir, "best_model.pth")
    ckpt_last = os.path.join(cfg.run_dir, "last_epoch.pth")
    hist_path = os.path.join(cfg.run_dir, "history.json")
    loss_png = os.path.join(cfg.run_dir, "loss_curves.png")
    acc_png = os.path.join(cfg.run_dir, "acc_curves.png")
    report_txt = os.path.join(cfg.run_dir, "test_report.txt")

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Datasets/Loaders
    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=eval_tf)
    test_ds = ImageFolder(test_dir, transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    print("Classes:", train_ds.classes)
    print("Sizes (train/val/test):", len(train_ds), len(val_ds), len(test_ds))

    # Model
    model = SimpleCNN(img_size=cfg.img_size, num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val = float("inf")

    print("\nTraining started...")
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(cur_lr)

        # Save last each epoch + history
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_val": best_val}, ckpt_last)
        save_json(history, hist_path)

        improved = early.step(va_loss)
        if improved:
            best_val = va_loss
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_val": best_val}, ckpt_best)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | lr={cur_lr:.2e} | {dt:.1f}s")

        if early.should_stop:
            print(f"Early stopping at epoch {epoch}. Best val_loss={best_val:.4f}")
            break

    # Plot curves
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(loss_png, dpi=200)

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(acc_png, dpi=200)

    print("\nSaved plots to:", cfg.run_dir)

    # Test with best checkpoint if exists, else last
    if os.path.exists(ckpt_best):
        ckpt = torch.load(ckpt_best, map_location=device)
        print("Loaded best checkpoint:", ckpt_best, "epoch:", ckpt["epoch"])
    else:
        ckpt = torch.load(ckpt_last, map_location=device)
        print("Loaded last checkpoint:", ckpt_last, "epoch:", ckpt["epoch"])

    model.load_state_dict(ckpt["model_state"])
    preds, labels = predict_all(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    rep = classification_report(labels, preds, target_names=test_ds.classes)
    cm = confusion_matrix(labels, preds)

    out_text = []
    out_text.append(f"Test Accuracy: {acc:.6f}")
    out_text.append("\nClassification Report:\n" + rep)
    out_text.append("\nConfusion Matrix:\n" + str(cm))

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out_text))

    print("\n" + "\n".join(out_text))
    print("\nSaved report to:", report_txt)
    print("Run folder:", os.path.abspath(cfg.run_dir))


if __name__ == "__main__":
    main()
