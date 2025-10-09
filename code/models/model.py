import torch
import torch.nn as nn
from .layers import *
import torch.nn.functional as F
from torch.distributions.normal import Normal


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow
    
class Dual_Unet(nn.Module):
    def __init__(self, in_c, n_filters, n_cls):
        super(Dual_Unet, self).__init__()
        self.cls = n_cls
        
        self.encoder = Encoder(in_c, n_filters)
        
        self.up4 = nn.ConvTranspose3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d4_1 = BasicConv3d(16 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.d4_2 = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        
        self.up3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d3_1 = BasicConv3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.d3_2 = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        
        self.up2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d2_1 = BasicConv3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.d2_2 = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        
        self.up1 = nn.ConvTranspose3d(2 * n_filters, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d1_1 = BasicConv3d(2 * n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.d1_2 = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        
        self.d0 = nn.Conv3d(n_filters, n_cls, kernel_size=3, stride=1, padding=1)
        
        self.fuse4 = Fusion(8 * n_filters)
        self.fuse3 = Fusion(4 * n_filters)
        self.fuse2 = Fusion(2 * n_filters)
        self.fuse1 = Fusion(n_filters)
        self.fuse0 = Fusion(n_cls)

        self.seg3 = BasicConv3d(4 * n_filters, 2 * n_filters, kernel_size=1, stride=1, padding=0)
        self.seg2 = BasicConv3d(2 * n_filters, n_filters, kernel_size=1, stride=1, padding=0)
        self.seg1 = BasicConv3d(n_filters, n_filters // 2, kernel_size=1, stride=1, padding=0)
        self.seg0 = BasicConv3d(n_filters // 2, n_cls, kernel_size=1, stride=1, padding=0)
        
        self.down = nn.Conv3d(n_filters // 2, n_cls, kernel_size=1, stride=1, padding=0)
        
    def forward(self, moving_img, fixed_img):
        m1, m2, m3, m4, m5 = self.encoder(moving_img)
        f1, f2, f3, f4, f5 = self.encoder(fixed_img)
        # print("m: ", m5.shape)
        
        dm4 = self.d4_2(self.d4_1(torch.cat([self.up4(m5), m4], dim=1)))
        dm3 = self.d3_2(self.d3_1(torch.cat([self.up3(dm4), m3], dim=1)))
        dm2 = self.d2_2(self.d2_1(torch.cat([self.up2(dm3), m2], dim=1)))
        dm1 = self.d1_2(self.d1_1(torch.cat([self.up1(dm2), m1], dim=1)))
        dm0 = self.d0(dm1)
        
        df4 = self.d4_2(self.d4_1(torch.cat([self.up4(f5), f4], dim=1)))
        df3 = self.d3_2(self.d3_1(torch.cat([self.up3(df4), f3], dim=1)))
        df2 = self.d2_2(self.d2_1(torch.cat([self.up2(df3), f2], dim=1)))
        df1 = self.d1_2(self.d1_1(torch.cat([self.up1(df2), f1], dim=1)))
        df0 = self.d0(df1)
        
        flow4 = self.fuse4(dm4, df4)
        flow3 = self.fuse3(dm3, df3, flow4)
        flow2 = self.fuse2(dm2, df2, flow3)
        flow1 = self.fuse1(dm1, df1, flow2)
        # print("flow1: ", flow1.shape)
        flow0 = self.fuse0(dm0, df0, self.down(flow1))


        flow3 = flow3 * self.seg3(F.interpolate(flow4, scale_factor=2, mode='trilinear', align_corners=True))
        # print("flow3: ", flow2.shape)
        flow2 = flow2 * self.seg2(F.interpolate(flow3, scale_factor=2, mode='trilinear', align_corners=True))
        # print("flow2: ", flow1.shape)
        flow1 = flow1 * self.seg1(F.interpolate(flow2, scale_factor=2, mode='trilinear', align_corners=True))
        # print("flow1: ", flow1.shape)
        flow = flow0 * self.seg0(flow1)
       
        return flow

class Encoder(nn.Module):
    def __init__(self, in_c, n_filters):
        super(Encoder, self).__init__()
        self.conv1_1 = BasicConv3d(in_c, n_filters, kernel_size=5, stride=1, padding=2)
        self.conv1_2 = BasicConv3d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        
        self.pool_1 = nn.MaxPool3d(2, 2)
        self.conv2_1 = BasicConv3d(n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = BasicConv3d(2 * n_filters, 2 * n_filters, kernel_size=3, stride=1, padding=1)
    
        
        self.pool_2 = nn.MaxPool3d(2, 2)
        self.conv3_1 = BasicConv3d(2 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = BasicConv3d(4 * n_filters, 4 * n_filters, kernel_size=3, stride=1, padding=1)
    
        self.pool_3 = nn.MaxPool3d(2, 2)
        self.conv4_1 = BasicConv3d(4 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = BasicConv3d(8 * n_filters, 8 * n_filters, kernel_size=3, stride=1, padding=1)
    
        self.pool_4 = nn.MaxPool3d(2, 2)
        self.conv5_1 = BasicConv3d(8 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = BasicConv3d(16 * n_filters, 16 * n_filters, kernel_size=3, stride=1, padding=1)

        self.da5 = nn.Sequential(DA_Block(16 * n_filters),
                                 DA_Block(16 * n_filters)
                                 )

    def forward(self, x):
        ds1 = self.conv1_2(self.conv1_1(x))  # n 160
        ds2 = self.conv2_2(self.conv2_1(self.pool_1(ds1)))  # 2n 80
        ds3 = self.conv3_2(self.conv3_1(self.pool_2(ds2)))  # 4n 40
        ds4 = self.conv4_2(self.conv4_1(self.pool_3(ds3)))  # 8n 20
        ds5 = self.conv5_2(self.conv5_1(self.pool_4(ds4)))  # 16 10
        # ds5 = self.da5(ds5)
        
        return ds1, ds2, ds3, ds4, ds5
        

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)
    
    
if __name__ == "__main__":
    x = torch.rand(2, 1, 160, 192, 160)
    tar = x
    # net = U_Network(3, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16])
    net = Dual_Unet(1, 16, 1)
    y = net(x, tar)
    print("y shape: ", y.shape)
    
