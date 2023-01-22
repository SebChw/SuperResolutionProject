from torch import nn

"""
Here we have very simple SR model
https://arxiv.org/abs/1501.00092
"""
class SRCNN(nn.Module):
    def __init__(self, kernel_sizes=[9, 1, 5], num_filters=[64, 32], model_parameters=None):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.model = nn.Sequential(
            nn.Conv2d(
                3, num_filters[0], kernel_size=kernel_sizes[0], padding="same"),
            nn.ReLU(),
            nn.Conv2d(num_filters[0], num_filters[1],
                      kernel_size=kernel_sizes[1]),
            nn.ReLU(),
            nn.Conv2d(num_filters[1], 3,
                      kernel_size=kernel_sizes[2], padding="same")
        )

    def forward(self, X):
        #! At this moment we assume that images have been put through bicubic interpolation
        #! and has the same size as expected result.
        return self.model(X)
