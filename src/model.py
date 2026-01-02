import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.backbone(x)
        return x.view(x.size(0), -1)


class MultiModalRegressor(nn.Module):
    def __init__(self, tabular_dim):
        super().__init__()

        self.image_encoder = ImageEncoder()

        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU()
        )

        self.regressor = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular):
        img_feat = self.image_encoder(image)
        tab_feat = self.tabular_net(tabular)
        x = torch.cat([img_feat, tab_feat], dim=1)
        return self.regressor(x).squeeze(1)
