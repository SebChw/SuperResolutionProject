from torch.utils.data import Dataset
import pandas as pd
import torchvision
from torch.nn.functional import interpolate
import torch
from utils import collect_paths
PREFIX = "DIV2K_"

class SRDataset(Dataset):
    def __init__(self,  scaling_factors = [2,3,4], downscalings = ["unknown"], train=True, transform = None, return_scaling_factor = True, data_path="data", perform_bicubic=False):
        self.scaling_factors = scaling_factors
        self.downscalings = downscalings
        self.train = train
        self.transform = transform
        self.return_scaling_factor = return_scaling_factor
        self.data_path = data_path

        self.perform_bicubic = perform_bicubic

        self.data_df = self.collect_paths()


    def collect_paths(self):
        prefix = PREFIX + ("train_" if self.train else "valid_")
        return collect_paths(self.data_path, prefix, self.downscalings, self.scaling_factors, patches="_patches")

    def __len__(self):
        return self.data_df.shape[0]
    
    def __getitem__(self, idx):
        input_path, target_path, scaling = self.data_df.iloc[idx]
    
        input_img = torchvision.io.read_image(str(input_path)).to(torch.float32)
        #! target images have different sizes within themselves
        target_img = torchvision.io.read_image(str(target_path)).to(torch.float32)

        if self.perform_bicubic:
            input_img = interpolate(input_img.unsqueeze(0), size = target_img.shape[1:])

        return input_img.squeeze(0), target_img