import torch
import torch.nn as nn
import torch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=False):
        super(SimpleCNN, self).__init__()

        simple_features = [
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]

        normal_classifier = [
            nn.Linear(16 * 112 * 112, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        ]

        dropout_features = [
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        ]

        dropout_classifier = [
            nn.Linear(16 * 112 * 112, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        ]

        self.features = nn.Sequential(
            *(simple_features if not dropout else dropout_features)
        )
        self.classifier = nn.Sequential(
            *(normal_classifier if not dropout else dropout_classifier)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

