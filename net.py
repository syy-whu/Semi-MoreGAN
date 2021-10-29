import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
import math
from torch.autograd import Variable
from modules import Residual_Block, FcaLayer,DetailedResidualBlock,PDNM,Constage

class GenerativeNetwork(nn.Module):
    def __init__(self, num_features=64):
        super(GenerativeNetwork, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False
        ####depth-attention prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            Residual_Block(32, 32),
            FcaLayer(32, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            Residual_Block(64, 64),
            FcaLayer(64, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            Residual_Block(128, 128),
            FcaLayer(128, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            Residual_Block(256, 256),
            FcaLayer(256, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            Residual_Block(256, 256),
            FcaLayer(256, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            Residual_Block(256, 256),
            FcaLayer(256, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            Residual_Block(256, 256),
            FcaLayer(256, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.SELU(inplace=True)
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            Residual_Block(128, 128),
            FcaLayer(128, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.SELU(inplace=True)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            Residual_Block(64, 64),
            FcaLayer(64, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=64),
            nn.SELU(inplace=True)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            Residual_Block(32, 32),
            FcaLayer(32, 8, 32, 32),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=32),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        ####depth-attention prediction network
        ####Detail recovery autoencoder
        self.autoencoder_head=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU(),
        )
        self.autoencoder_body = nn.Sequential(
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features),
            DetailedResidualBlock(num_features)
            # DetailedResidualBlock(num_features),
            # DetailedResidualBlock(num_features),
            # DetailedResidualBlock(num_features)
        )
        # self.U_net = nn.Sequential(
        #     U_Net()
        # )
        ####Detail recovery autoencoder
        self.dgnlb = PDNM(num_features)
        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self,x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)
        ################################## rain removal
        f = self.autoencoder_head(x)
        f = self.autoencoder_body(f)
        # f = self.U_net(f)
        f = self.dgnlb(f, depth_pred.detach())
        r = self.tail(f)
        x = x + r
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        if self.training:
            return x, depth_pred
            # return x, depth_pred

        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # 4*Conv layers coupled with leaky-relu & instance norm.
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

if __name__ == '__main__':
    ds_net = Discriminator(512,1024).cuda().train()
    gs_net = GenerativeNetwork().cuda().train()
    num=0
    MSE_criterion = nn.MSELoss()
    DS_optimizer = optim.Adam(ds_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    while num<1000:

        input=torch.randn(1,3,512,1024)
        lable=torch.randn(1,3,512,1024)
        input=Variable(input).cuda()
        lable = Variable(lable).cuda()
        GS_result = gs_net(input)