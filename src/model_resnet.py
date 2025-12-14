import torch.nn as nn
from torchvision import models

def get_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1")

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2)
    )

    return model
