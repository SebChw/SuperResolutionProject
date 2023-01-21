from src.sr_dataset import SRDataset
from src.edsr import get_edsr
import argparse
import yaml
import torch
import cv2
from PIL import Image
from torchvision import transforms

def inference(config):
    model = get_edsr(config['model_parameters'])
    model = model.load_from_checkpoint(
        f"trained_models/{config['architecture']}.ckpt", strict=False)
    model.eval()
    
    img = Image.open("data/DIV2K_valid_LR_unknown/X2/0807x2.png")
    pil_images = [img]  # batch size is one
    transform = transforms.Compose([transforms.ToTensor()])
    x = torch.cat(
        [transform(i).unsqueeze(0) for i in pil_images])
    new = model(x)
    cv2.imshow("new", new.permute(0, 2, 3, 1).squeeze().detach().numpy())
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='Runs training on a given architecture',
    )
    parser.add_argument('-c', '--config_path',
                        default='configs/train_edsr.yaml', required=False)
    args = parser.parse_args()
    with open(args.config_path, "r") as config:
        inference(yaml.safe_load(config))
