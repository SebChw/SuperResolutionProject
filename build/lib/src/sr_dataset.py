from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import torchvision
from torch.nn.functional import interpolate
import torch
import numpy as np
from torchsr.utils import collect_paths

PREFIX = "DIV2K_"


class SRDataset(Dataset):
    def __init__(self,  scaling_factors=[2, 3, 4], downscalings=["unknown"], train=True,
                 data_path="data", bicubic_down=False, bicubic_up=False,
                 normalize=True, patches="", extension="png"):

        self.scaling_factors = scaling_factors
        self.downscalings = downscalings
        self.train = train
        self.data_path = data_path
        self.normalize = normalize
        self.extension = extension

        self.bicubic_down = bicubic_down
        self.bicubic_up = bicubic_up

        self.patches = patches
        self.data_df = self.collect_paths()

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

        target_img = self.load_image(target_path)

        if self.bicubic_down:
            size = np.array(target_img.shape[1:]) // scaling
            input_img = interpolate(
                target_img.unsqueeze(0), size=(size[0], size[1]), mode='bicubic')
        else:
            input_img = self.load_image(input_path).unsqueeze(0)

        if self.bicubic_up:
            input_img = interpolate(
                input_img, size=target_img.shape[1:], mode="bicubic")

        if self.normalize:
            input_img /= 255
            target_img /= 255

        return input_img.squeeze(0), target_img
