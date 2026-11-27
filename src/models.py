import torch.nn as nn


def block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CryCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128)
        )
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))
