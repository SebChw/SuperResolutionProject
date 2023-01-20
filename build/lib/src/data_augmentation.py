from kornia.augmentation import RandomVerticalFlip, RandomHorizontalFlip, RandomRotation
import torch.nn as nn
import torch
from torch import Tensor


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        #It's not specified as sequential as we must apply same transformations to input and outputs!
        self.ts = [
            RandomHorizontalFlip(p=0.1),
            RandomVerticalFlip(p=0.1),
            RandomRotation(degrees=15.0, p=0.1)
        ]

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x_out = self.ts[2](self.ts[1](self.ts[0](x)))
        #! by passing _params y is transformed in exactly the same way as x
        y_out = self.ts[2](self.ts[1](self.ts[0](y, self.ts[0]._params), self.ts[1]._params), self.ts[2]._params)
        return x_out, y_out


if __name__ == '__main__':
    aug = DataAugmentation()
    batch = torch.randn((32, 3, 128,128))
    x, y = aug(batch, batch)
    assert (x == y).all()
