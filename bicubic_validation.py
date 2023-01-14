from torchmetrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from sr_dataset import SRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import neptune.new as neptune


"""This is just to see what results can be obtained using just bicubic interpolation

    BICUBIC INTERPOLATION IS NOT INVERTIBLE OPERATION X -> bicubic downscale -> bicubic upscale won't give you exact X back!
"""
#! you must have your API token and project name configured for this to work!
run = neptune.init_run()
run["algorithm"] = "BicubicInterpolation"

params = {
    "perform_bicubic":True,
    "scaling_factors":[2,3,4], 
    "downscalings": ["unknown"], 
    "train":False
}

run["data/parameters"] = params
run["data_versions/valid"].track_files("data.dvc")

data_set = SRDataset(**params)
loader = DataLoader(data_set, batch_size=1)

psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure()

for batch in tqdm(loader):
    x,y = batch
    psnr(x,y)
    ssim(x,y)

final_psnr = psnr.compute()
final_ssim = ssim.compute()

print(final_psnr, final_ssim)

run["psnr"] = final_psnr
run["ssim"] = final_ssim

run.stop()