import torch
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.numpy())

    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))
