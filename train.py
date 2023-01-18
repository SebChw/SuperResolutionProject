import sys
import getopt
import argparse
import yaml

from srcnn import LitSRCNN
from sr_dataset import SRDataset

from torch.utils.data import DataLoader
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from custom_callbacks import ImageLoggingCallback

# from neptune.new import ANONYMOUS_API_TOKEN


class Trainer:
    def __init__(self):
        self.architecture = LitSRCNN
        self.help = '''
                    -a, --architecture: architecture to use, e.g. srcnn 
                    -h, --help: help
                    '''
        self.neptune_logger = NeptuneLogger(
            # api_key=ANONYMOUS_API_TOKEN,
            project="skdbmk/superresolution",
            tags=["training", self.architecture.__name__],  # optional
        )

    def run(self, config):
        self.parse_args(config)
        train_loader, val_loader = self.get_data()
        model = self.architecture()
        trainer = pl.Trainer(accelerator="gpu", devices=1,
                             precision=16, logger=self.neptune_logger, callbacks=[
                                 ImageLoggingCallback()], default_root_dir=f"model_checkpoints/{self.architecture.__name__}", max_epochs=5)
        trainer.fit(model, train_loader, val_loader)
        # input_sample = torch.randn((192,192,3))
        # model.to_onnx(
        #     f"model_{self.architecture.__name__}.onnx", input_sample, export_params=True)
        # TODO take a few random images from traindataset and create before -> after    

    def parse_args(self, config):
        architecture_type = config["architecture"]
        self.batch_size = config["batch_size"]
        if architecture_type == 'srcnn':
            self.architecture = LitSRCNN
        else:
            print('Architecture not implemented')
            sys.exit()

    def get_data(self):
        trainset = SRDataset(return_scaling_factor=False,
                             perform_bicubic=True, patches="_patches192")
        validset = SRDataset(
            train=False, return_scaling_factor=False, perform_bicubic=True)

        # TODO I observed small GPU Utilization probably n_workers could be set automatically?
        train_loader = DataLoader(trainset, batch_size=32)
        # ! here we evaluate work on big images!
        val_loader = DataLoader(validset, batch_size=1)
        return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='Runs training on a given architecture',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/training.yaml', required=False)
    args = parser.parse_args()
    Trainer = Trainer()
    with open(args.config_path, "r") as config:
        Trainer.run(yaml.safe_load(config))
