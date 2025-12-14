import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

def train_model(model, train_loader, val_loader, device, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                out = model(x.to(device))
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.numpy())

        acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
