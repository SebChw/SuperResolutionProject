from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import torchvision
import cv2
from torch.nn.functional import interpolate
import torch
import numpy as np
from utils import collect_paths
PREFIX = "DIV2K_"


class SRDataset(Dataset):
    def __init__(self,  scaling_factors=[2, 3, 4], downscalings=["unknown"], train=True,
                 transform=None, return_scaling_factor=True, data_path="data", perform_bicubic=False,
                 normalize=True, patches="", extension="png", random_dataset_order=False):

        self.scaling_factors = scaling_factors
        self.downscalings = downscalings
        self.train = train
        self.transform = transform
        self.return_scaling_factor = return_scaling_factor
        self.data_path = data_path
        self.normalize = normalize
        self.extension = extension
        self.random_dataset_order = random_dataset_order

        self.perform_bicubic = perform_bicubic
        self.patches = patches
        self.data_df = self.collect_paths()

        if self.random_dataset_order:
            self.data_df = self.data_df.sample(frac=1)

        self.data_df = self.data_df.to_numpy()

    def collect_paths(self):
        prefix = PREFIX + ("train_" if self.train else "valid_")
        return collect_paths(self.data_path, prefix, self.downscalings,
                             self.scaling_factors, patches=self.patches,
                             extension=self.extension)

    def __len__(self):
        return self.data_df.shape[0]

    def load_image(self, path):
        if self.extension == "png":
            return torchvision.io.read_image(str(path)).to(torch.float32)
        else:
            return torch.from_numpy(np.load(path)).to(torch.float32)

    def __getitem__(self, idx):
        input_path, target_path, scaling = self.data_df[idx]

        input_img, target_img = self.load_image(
            input_path), self.load_image(target_path)

        if self.perform_bicubic:
            input_img = interpolate(
                input_img.unsqueeze(0), size=target_img.shape[1:])

        if self.normalize:
            input_img /= 255
            target_img /= 255

        return input_img.squeeze(0), target_img
