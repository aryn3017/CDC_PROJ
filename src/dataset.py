import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class PropertyDataset(Dataset):
    def __init__(self, data_path, image_dir, features, scaler=None, train=True):
        self.df = (
            pd.read_csv(data_path)
            if data_path.endswith(".csv")
            else pd.read_excel(data_path)
        )
        self.image_dir = image_dir
        self.features = features
        self.scaler = scaler
        self.train = train

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        img_path = os.path.join(self.image_dir, f"{row['id']}.png")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tabular
        tabular = row[self.features].values.astype(float)
        if self.scaler is not None:
            tabular = self.scaler.transform(tabular.reshape(1, -1))[0]
        tabular = torch.tensor(tabular, dtype=torch.float32)

        if self.train:
            target = torch.tensor(
                np.log1p(row["price"]),
                dtype=torch.float32
            )
            return image, tabular, target

        return image, tabular
