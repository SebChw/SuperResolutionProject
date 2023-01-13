from torch.nn.functional import interpolate
from torchmetrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from sr_dataset import SRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

data_set = SRDataset(perform_bicubic=True, scaling_factors=[2], downscalings = ["bicubic"], train=False)
loader = DataLoader(data_set, batch_size=1)

psnr = PeakSignalNoiseRatio()
ssim = StructuralSimilarityIndexMeasure()

for batch in tqdm(loader):
    x,y = batch
    psnr(x,y)
    ssim(x,y)

print(psnr.compute())
print(ssim.compute())