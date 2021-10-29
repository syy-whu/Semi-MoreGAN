import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import scipy.stats as st
from torch.nn import Softmax


class Constage(nn.Module):
    def __init__(self, input_dims,out_dims,k_size, stride,padding=1):
        super(Constage, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dims, out_dims, k_size, stride, padding),
            nn.SELU(),
        )

    def forward(self, x):
        output = self.layers(x)

        return output
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)
def get_dct_weights(width, height, channel, fidx_u=[0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3], fidx_v=[0, 1, 0, 5,
                                                                                                             2, 0, 2, 0,
                                                                                                             0, 6, 0, 4,
                                                                                                             6, 3, 2,
                                                                                                             5]):
    # width : width of input
    # height : height of input
    # channel : channel of input
    # fidx_u : horizontal indices of selected fequency
    # according to the paper, should be [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3]
    # fidx_v : vertical indices of selected fequency
    # according to the paper, should be [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]
    # [0,0],[0,1],[6,0],[0,5],[0,2],[1,0],[1,2],[4,0],
    # [5,0],[1,6],[3,0],[0,4],[0,6],[0,3],[2,2],[3,5],
    scale_ratio = width // 7
    fidx_u = [u * scale_ratio for u in fidx_u]
    fidx_v = [v * scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    # split channel for multi-spectal attention
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * c_part: (i + 1) * c_part, t_x, t_y] \
                    = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    # Eq. 7 in our paper
    return dct_weights

class FcaLayer(nn.Module):
    def __init__(self, channel, reduction, width, height):
        super(FcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, channel))
        # self.register_parameter('pre_computed_dct_weights',torch.nn.Parameter(get_dct_weights(width,height,channel)))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (self.height, self.width))
        s= y * self.pre_computed_dct_weights
        y = torch.sum(y * self.pre_computed_dct_weights, dim=(2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1, downsample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=o_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class  DetailedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(DetailedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1), nn.ReLU()
        )
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv1 = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=3, dilation=3)
        self.conv3 = nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=5, dilation=5)
        self.conv4 = nn.Sequential(nn.Conv2d(channels*3, channels, (3, 3), stride=(1, 1), padding=1),
                                   Residual_Block(channels,channels),
                                   nn.ReLU())
    def forward(self, x):
        inputs = self.conv0(x)
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        catout = torch.cat((x1, x2, x3), 1)
        out = self.conv4(catout)
        return x + out
        # return x + conv1
class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.out = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels,kernel_size=1, stride=1)


    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, int(h/pool_size), int(h/pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)
        # final = torch.cat(output_slices, dim=1)
        return self.out(torch.cat(output_slices, dim=1))


class PDNM(nn.Module):
    def __init__(self, in_channels):
        super(PDNM, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)
        self.pyramid_pooling_g = pyramidPooling(int(in_channels / 2), [8, 4, 2, 1])
        self.pyramid_pooling_the = pyramidPooling(int(in_channels / 2), [8, 4, 2, 1])
        self.pyramid_pooling_phi = pyramidPooling(int(in_channels / 2), [8, 4, 2, 1])

    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        x_down = self.down(x)
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2)
        g = self.pyramid_pooling_g(g).view(n, int(c / 2), -1).transpose(1, 2)
        theta = self.theta(x_down)
        theta = self.pyramid_pooling_the(theta).view(n, int(c / 2), -1).transpose(1, 2)
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2)
        phi = self.pyramid_pooling_phi(phi).view(n, int(c / 2), -1)
        Ra = F.softmax(torch.bmm(theta, phi), 2)
        depth1 = F.interpolate(depth_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners = True).view(n, 1, int(h / 4)*int(w / 4)).transpose(1,2)
        depth2 = F.interpolate(depth_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners = True).view(n, 1, int(h / 8)*int(w / 8))
        depth1_expand = depth1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        depth2_expand = depth2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        Rd = torch.min(depth1_expand / (depth2_expand + self.eps), depth2_expand / (depth1_expand + self.eps))
        Rd = F.softmax(Rd, 2)
        S = F.softmax(Ra * Rd, 2)
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))
        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)
