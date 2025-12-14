import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_tf)

    total = len(full_dataset)
    train_len = int(0.7 * total)
    val_len = int(0.15 * total)
    test_len = total - train_len - val_len

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_len, val_len, test_len]
    )

    val_ds.dataset.transform = test_tf
    test_ds.dataset.transform = test_tf

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )
