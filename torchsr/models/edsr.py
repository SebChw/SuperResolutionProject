import torch
import torch.nn as nn
import math


"""Implementation inspired from https://github.com/sanghyun-son/EDSR-PyTorch"""

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        #! For deeper models they've used res_scale of 0.1
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n? nice trick
            # every time I double the size of the image
            for _ in range(int(math.log(scale, 2))):
                # r^2 = 4 => r=2, upscale factor
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))  # This we do in just a one shot
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class EDSR(nn.Module):
    def __init__(self, model_parameters, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = model_parameters['n_resblocks']
        n_feats = model_parameters['n_feats']
        kernel_size = 3
        scale = model_parameters['scaling_factors'][0]
        act = nn.ReLU(inplace=True)

        n_colors = 3

        self.sub_mean = MeanShift(model_parameters['rgb_range'])
        self.add_mean = MeanShift(model_parameters['rgb_range'], sign=1)

        # define head module
        m_head = nn.ModuleList([conv(n_colors, n_feats, kernel_size)])

        # define body module
        m_body = nn.ModuleList([
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=model_parameters['res_scale']
            ) for _ in range(n_resblocks)
        ])
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = nn.ModuleList([
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ])

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x


if __name__ == '__main__':
    model_parameters = {
        'rgb_range': 255,
        'scaling_factors': [2],
        'n_resblocks': 16,
        'n_feats': 64,
        'res_scale': 1
    }

    esr = EDSR(model_parameters)

    img = torch.randint(0, 255, (1, 3, 192, 192), dtype=torch.float32)
    esr(img)
