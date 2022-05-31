import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchstat import stat

class ContextualBlock(nn.Module):
    def __init__(self, channels):
        super(ContextualBlock, self).__init__()
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
class PDNL(nn.Module):
    def __init__(self, in_channels):
        super(PDNL, self).__init__()

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
        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners = True)

class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out + input

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
        self.upv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
        )

        self.upv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.upv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.upv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.self_attention = nn.Sequential(
            selfattention(256)
        )
        self.downv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.downv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.downv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
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
            ContextualBlock(num_features),
            ContextualBlock(num_features),
            ContextualBlock(num_features),
            ContextualBlock(num_features)
        )

        ####Detail recovery autoencoder
        self.dgnlb = PDNL(num_features)
        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self,x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.upv1(x)
        d_f2 = self.upv2(d_f1)
        d_f3 = self.upv3(d_f2)
        d_f4 = self.upv4(d_f3)
        d_f5 = self.self_attention(d_f4)
        d_f6 = self.downv1(d_f5)
        d_f7 = self.downv2(d_f6+d_f3)
        d_f8 = self.downv3(d_f7+d_f2)
        depth_pred = self.depth_pred(d_f8 + d_f1)
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

        return x,depth_pred


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
    # ds_net = Discriminator(512,1024).cuda().train()
    gs_net = GenerativeNetwork()
    num=0
    MSE_criterion = nn.MSELoss()
    DS_optimizer = optim.Adam(gs_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    while num<1:

        input=torch.randn(1,3,1000,1024)
        lable=torch.randn(1,3,512,1024)
        input=Variable(input).cuda()
        lable = Variable(lable).cuda()
        stat(gs_net, (3, 512,1024))
        num=num+1
        # GS_result = gs_net(input)