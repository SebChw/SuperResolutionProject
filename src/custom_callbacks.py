from pytorch_lightning.callbacks import Callback
from sr_dataset import SRDataset
import torch
import torchvision
from pathlib import Path

from utils import cut_tensor_from_0_to_1
"""All callbacks here assume that logger is a neptune one"""


class ImageLoggingCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.img_num = 0

        self.validset = SRDataset(
            train=False, **config['dataset_parameters'])

    def on_validation_epoch_end(self, trainer, pl_module):
        # We always take the same photo and take only one so that neptun doesn't cry
        # Not the most efficient solution
        # 6th example from validation set seems to be hard
        lr, hr = self.validset[6]

        # making it batch of size 1, it must be on the same device as model
        lr_to_hr = pl_module(torch.unsqueeze(lr, 0).to("cuda"))[0]

        # it is probably the other way around as in opencv
        lr_filled = torch.zeros((3, lr.shape[1], hr.shape[2]))
        lr_filled[:, :, :hr.shape[2]] = lr

        lr_to_hr = cut_tensor_from_0_to_1(lr_to_hr)

        grid = torchvision.utils.make_grid(torch.cat([lr_filled, lr_to_hr.to(
            "cpu"), hr], axis=2))  # lr_to_hr must be on the same device as the rest

        # TODO Save the images in some logging folder
        img_path = f"img{self.img_num}.png"
        torchvision.utils.save_image(grid, img_path)
        pl_module.logger.experiment[f"validation/results{self.img_num}"].upload(
            img_path)
        self.img_num += 1

        #! If i remove this image here it won't be sent to neptune :(
        # Path(img_path).unlink()
