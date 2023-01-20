from src.sr_dataset import SRDataset
from src.edsr import get_edsr
import argparse
import yaml
import torch
import cv2


def create_onnx(config):
    model = get_edsr(config['model_parameters'])
    model = model.load_from_checkpoint(
        f"trained_models/{config['architecture']}.ckpt", strict=False)
    model.eval()
    data = SRDataset(scaling_factors=[2], train=False, data_path="data")
    x, y = data[0]
    new = model(x.unsqueeze(0))
    cv2.imshow("x", x.permute(1, 2, 0).numpy())
    cv2.imshow("y", y.permute(1, 2, 0).numpy())
    cv2.imshow("new", new.permute(0, 2, 3, 1).squeeze().detach().numpy())
    cv2.waitKey(0)
    # input_sample = torch.randn((3, 192, 192))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='Runs training on a given architecture',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/train_edsr.yaml', required=False)
    args = parser.parse_args()
    with open(args.config_path, "r") as config:
        create_onnx(yaml.safe_load(config))
