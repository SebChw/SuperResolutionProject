import sys
import getopt

from srcnn import LitSRCNN
from sr_dataset import SRDataset

from torch.utils.data import DataLoader

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

    def run(self, argv):
        self.parse_args(argv)
        train_loader, val_loader = self.get_data()
        model = self.architecture()
        trainer = pl.Trainer(accelerator="gpu", devices=1,
                             precision=16, logger=self.neptune_logger, callbacks=[
                                 ImageLoggingCallback()], default_root_dir=f"model_checkpoints/{self.architecture.__name__}")
        trainer.fit(model, train_loader, val_loader)
        # model.model.set_swish(memory_efficient=False)
        # model.to_onnx(
        #     f"model_{self.architecture.__name__}.onnx", export_params=True)
        # TODO take a few random images from traindataset and create before -> after    

    def parse_args(self, argv):
        opts, _ = getopt.getopt(argv, shortopts="ha:", longopts=[
                                "help", "architecture="])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print(self.help)
                sys.exit()
            elif opt in ("-a", "--architecture"):
                architecture_type = arg
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
    Trainer = Trainer()
    Trainer.run(sys.argv[1:])
