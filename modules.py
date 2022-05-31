import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import scipy.stats as st
from torch.nn import Softmax




class DetailedResidualBlock(nn.Module):
    def __init__(self, channels):
        super(DetailedResidualBlock, self).__init__()
        self.orginal_conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1),
                                          nn.BatchNorm2d(channels),
                                          nn.ReLU())
        self.deconv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=1, dilation=1),
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=3, dilation=3),
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=5, dilation=5)
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=1, dilation=1),
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=3, dilation=3),
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=5, dilation=5)
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=1, dilation=1),
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=3, dilation=3),
            nn.Conv2d(channels, channels, (3, 3), stride=(1, 1), padding=5, dilation=5)
        )
        self.conv_c = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.recon = nn.Sequential(nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(channels),
                                   nn.ReLU())

    def forward(self, x):
        inputs = self.orginal_conv(x)
        x1 = self.deconv1(inputs)
        x2 = self.deconv2(inputs+x1)
        x3 = self.deconv3(inputs+x2)
        catout = torch.cat((x1, x2, x3), 1)
        out = self.conv_c(catout)
        out = self.recon(inputs + out)
        return out

# class DetailedResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(DetailedResidualBlock, self).__init__()
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(channels, int(channels/2), kernel_size=1), nn.ReLU()
#         )
#         # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
#         self.conv1 = nn.Conv2d(int(channels/2), int(channels/2), (3, 3), stride=(1, 1), padding=1)
#         self.conv2 = nn.Conv2d(int(channels/2), int(channels/2), (3, 3), stride=(1, 1), padding=3, dilation=3)
#         self.conv3 = nn.Conv2d(int(channels/2), int(channels/2), (3, 3), stride=(1, 1), padding=5, dilation=5)
#         self.conv4 = nn.Sequential(nn.Conv2d(int(channels/2*3), channels, kernel_size=1),
#                                    # Residual_Block(channels,channels),
#                                    nn.BatchNorm2d(channels),
#                                    nn.ReLU())
#
#     def forward(self, x):
#         inputs = self.conv0(x)
#
#         x1 = self.conv1(inputs)
#         x2 = self.conv2(inputs)
#         x3 = self.conv3(inputs)
#         catout = torch.cat((x1, x2, x3), 1)
#         out = self.conv4(catout)
#         return x + out

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
class PAPAModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1,4, 8, 16), dimension=2):
        super(PAPAModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        #size = _pair(size)
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats, atten):
        n, c, _, _ = feats.size()
        priors = []
        for indx, stage in enumerate(self.stages):
            if atten is not None:
                prior = stage(feats * atten[:,indx:indx+1,:,:]).view(n, c, -1)
            else:
                prior = stage(feats).view(n, c, -1)
            priors.append(prior)

        center = torch.cat(priors, -1)
        return center

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
            # out = nn.AdaptiveAvgPool2d(x, int(h / pool_size), int(h / pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear')
            output_slices.append(out)
        # final = torch.cat(output_slices, dim=1)
        return self.out(torch.cat(output_slices, dim=1))


class PDNM1(nn.Module):
    def __init__(self, in_channels):
        super(PDNM1, self).__init__()

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

class PDNM(nn.Module):
    def __init__(self, in_channels):
        super(PDNM, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.attention = nn.Conv2d(in_channels, 4, kernel_size=1)
        self.pyramid_pooling_phi = PAPAModule()
        self.pyramid_pooling_g = PAPAModule()
        self.depth_down1 = nn.Conv1d(2048,337,kernel_size=1)
        self.depth_down2 = nn.Conv1d(2048, 337, kernel_size=1)



        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)


    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        ###depth
        depth1 = F.interpolate(depth_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(n, 1,
                                                                                                                   int(
                                                                                                                       h / 4) * int(
                                                                                                                       w / 4)).transpose(
            1, 2)
        depth2 = F.interpolate(depth_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(n, 1,
                                                                                                                   int(
                                                                                                                       h / 8) * int(
                                                                                                                       w / 8))
        depth1_expand = self.depth_down1(depth1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8)).transpose(1, 2)).transpose(1, 2)
        depth2_expand = self.depth_down2(depth2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8)).transpose(1, 2)).transpose(1, 2)
        Rd = torch.min(depth1_expand / (depth2_expand + self.eps), depth2_expand / (depth1_expand + self.eps))
        Rd = F.softmax(Rd, 2)
        ###
        x_down = self.down(x)
        attention = self.attention(x_down)
        g = self.g(x_down)
        g = self.pyramid_pooling_g(g,attention).transpose(1, 2)
        phi = self.phi(x_down)
        phi = self.pyramid_pooling_phi(phi,attention)
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        Ra = F.softmax(torch.bmm(theta, phi), 2)
        S = F.softmax(Ra * Rd, 2)
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))


        # g = self.pyramid_pooling_g(g).view(n, int(c / 2), -1).transpose(1, 2)
        #
        # theta = self.theta(x_down)
        # theta = self.pyramid_pooling_the(theta).view(n, int(c / 2), -1).transpose(1, 2)
        #
        # phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2)
        # phi = self.pyramid_pooling_phi(phi).view(n, int(c / 2), -1)
        #
        #
        # Ra = F.softmax(torch.bmm(theta, phi), 2)
        # depth1 = F.interpolate(depth_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners = True).view(n, 1, int(h / 4)*int(w / 4)).transpose(1,2)
        # depth2 = F.interpolate(depth_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners = True).view(n, 1, int(h / 8)*int(w / 8))
        # depth1_expand = depth1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # depth2_expand = depth2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # Rd = torch.min(depth1_expand / (depth2_expand + self.eps), depth2_expand / (depth1_expand + self.eps))
        # Rd = F.softmax(Rd, 2)
        # S = F.softmax(Ra * Rd, 2)
        # y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))
        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)