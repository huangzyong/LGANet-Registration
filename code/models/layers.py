import torch
import torch.nn as nn
import torch.nn.functional as F
# from pivit import BasicSwinLayer


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv3d, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, bias=True, **kwargs)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.GELU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.norm(x1)
        x3 = self.relu(x2)
        if self.in_c == self.out_c:
            x3 = x3 + x
        return x3

class ResNet(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU())
        
    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        
        return x + res
        

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)  # (1, c)
        y_max = self.max_pool(x).view(b, c)  # (1, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)

        return self.sigmoid(y_avg+y_max)  # (1, c, 1, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.cat([avg_out, max_out], dim=1)
        w = self.conv1(w)
        return self.sigmoid(w) * x 

    
class ConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=2, **kwargs):
        super(ConvNeXt, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels * reduction, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm3d(out_channels * reduction),
            nn.Conv3d(out_channels * reduction, out_channels, kernel_size=1, stride=1),
            nn.GELU()
        )

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x0 = self.conv(x)
        if self.in_c == self.out_c:
            out = x + x0
        else:
            out = x0 + self.conv1(x)
        return out
    

class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*depth).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*depth)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*depth)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, depth)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, depth)

        out = self.gamma*out + x
        return out


class DA_Block(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(DA_Block, self).__init__()
        self.chanel_in = in_dim
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)

    def forward(self,x):
        x1 = self.pam(x)
        x2 = self.cam(x)
        return x1+x2
        # return x2
    
class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.out_c = channel if channel>1 else 2
        self.conv = BasicConv3d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.res = BasicConv3d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv3d(channel, self.out_c//2, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x_m, x_f, flow=None):
        if flow is None:
            flow = nn.Parameter(torch.ones_like(x_m)).to(x_m.device)
        elif x_m.shape[-1]>flow.shape[-1]:
            flow = F.interpolate(flow, scale_factor=2, mode='trilinear', align_corners=True)
        else:
            pass
        x_w = x_m * flow
        # x_d = x_f - x_m  # correlation
        x = torch.cat([x_w, x_f], dim=1)
        out = self.conv(x)
        out = self.res(out) + out
        out = self.conv1(out)
        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(nn.AvgPool3d(kernel_size=5, stride=2, padding=2),
                                    nn.InstanceNorm3d(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool3d(kernel_size=9, stride=4, padding=4),
                                    nn.InstanceNorm3d(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool3d(kernel_size=7, stride=8, padding=3),
                                    nn.InstanceNorm3d(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool3d((3, 1, 1)),
                                    nn.InstanceNorm3d(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    nn.InstanceNorm3d(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    nn.InstanceNorm3d(branch_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    nn.InstanceNorm3d(branch_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    nn.InstanceNorm3d(branch_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    nn.InstanceNorm3d(branch_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    nn.InstanceNorm3d(branch_planes * 5),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    nn.InstanceNorm3d(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]  
        depth = x.shape[-3]       
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[depth, height, width],
                        mode='trilinear')+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[depth, height, width],
                        mode='trilinear')+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[depth, height, width],
                        mode='trilinear')+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[depth, height, width],
                        mode='trilinear')+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out   

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Correlation3D(nn.Module):
    """
    Main model
    """
    def __init__(self,in_channel, kernel_size=3, d=3, sw=1, sf=2):
        super(Correlation3D, self).__init__()
        self.kernel_size = kernel_size
        self.block = nn.Sequential()
        self.d = d
        self.sw = sw
        self.sf = sf
        self.w = torch.ones((in_channel, 1, self.kernel_size, self.kernel_size, self.kernel_size)).cuda(1)

    def forward(self, mov, fix):
        B, C, H, W, T = mov.shape

        pm = F.conv3d(mov, self.w, stride=self.sw, padding=1, groups=C) # H
        pf = F.conv3d(fix, self.w, stride=self.sw, padding=self.sf+1, groups=C)  # H+4

        concat = []
        for i in range(self.d):
            for j in range(self.d):
                for k in range(self.d):
                    pf_crop = pf[:, :, i*self.sf:(i*self.sf+H), j*self.sf:(j*self.sf+W), k*self.sf:(k*self.sf+T)]
                    concat.append(torch.sum(pm*pf_crop, dim=1, keepdim=True))

        corr = torch.cat(concat, dim=1)/self.kernel_size**3
        return corr
    
from torch.distributions.normal import Normal

class PRplusplusBlock(nn.Module):
    def __init__(self, size, in_channel, in_flow=True, scale=True, kernel_size=3, d=3, sw=1, sf=2):
        super(PRplusplusBlock, self).__init__()
        self.scale = scale
        self.in_flow = in_flow
        if in_flow:
            self.stn = SpatialTransformer(size)
            # if scale:
            #     self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.corr = Correlation3D(in_channel, kernel_size, d, sw, sf)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel*2+kernel_size**3, in_channel*2+kernel_size**3, 3, 1, 1),
            nn.Conv3d(in_channel * 2 + kernel_size ** 3, in_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.Conv3d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU()
        )
        self.flow = nn.Conv3d(in_channel, 3, 3, 1, 1)
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

    def forward(self, x, y, flow=None):
        if self.in_flow:
            # if self.scale:
            #     flow = self.upsample(flow*2)
            x = self.stn(x, flow)
        corr = self.corr(x, y)

        stack = torch.cat([x, corr, y], dim=1)
        x = self.conv1(stack)
        res = self.conv2(x)
        flow = self.flow(x+res)
        return flow
    
    
class SwinBlock(nn.Module):
    def __init__(self, dim=16, depth=1, num_heads=1, window_size=(128, 9, 11, 9)):
        super(SwinBlock, self).__init__()
        # self.swin1 = BasicSwinLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size[1:])
        self.attn = LGAttention(dim=dim, shape=window_size)
        # self.swin2 = BasicSwinLayer(dim=dim, depth=depth, num_heads=num_heads, window_size=window_size[1:])
        
    def forward(self, x):
        
        # x = self.swin1(x)
        x = self.attn(x)
        # x = self.swin2(s1)
        
        return x
    
    
class FuseBlock(nn.Module):
    def __init__(self, in_c):
        super(FuseBlock, self).__init__()
        self.conv_q = BasicConv3d(in_c, in_c, kernel_size=3, stride=1, padding=1)
        self.conv_k = BasicConv3d(in_c, in_c, kernel_size=3, stride=1, padding=1)
        self.conv_v = BasicConv3d(in_c, in_c, kernel_size=3, stride=1, padding=1)
        self.lm = nn.LayerNorm
    
    def forward(self, x):
        q = self.conv_q(x)
        print("q: ", q.shape)
        k = self.conv_k(x)
        print("k: ", k.shape)
        v = self.conv_v(x)
        
        f = self.lm(q @ k.permute(0, 1, 2, 4, 3))
        
        
        return x

class DE_Block(nn.Module):
    def __init__(self, in_c):
        super(DE_Block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, 2 * in_c, 3, 1, 1),
            nn.InstanceNorm3d(2*in_c),
            nn.Conv3d(2 * in_c, 2 * in_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(2 * in_c, 4 * in_c, 3, 1, 1),
            nn.InstanceNorm3d(2*in_c),
            nn.Conv3d(4 * in_c, 4 * in_c, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(4 * in_c, 4 * in_c, 3, 1, 1),
            nn.Conv3d(4 * in_c, 4 * in_c, 3, 1, 1),
            nn.ReLU())
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(4 * in_c, 2 * in_c, 3, 1, 1),
            nn.Conv3d(2 * in_c, in_c, 3, 1, 1),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(4 * in_c, 2 * in_c, 3, 1, 1),
            nn.Conv3d(2 * in_c, 3, 3, 1, 1),
            nn.ReLU())
        
    def forward(self, x):
        s = self.conv(x)
        M = self.conv1(s)
        flow = self.conv2(s)
        return flow, M

class DABlock(nn.Module):
    def __init__(self, size, in_channel, in_flow=True, use_da=False, use_de=False, use_swin=False, window_size=(128, 9, 11, 9)):
        super(DABlock, self).__init__()
        self.in_flow = in_flow
        self.use_da = use_da
        self.use_de = use_de
        self.use_swin = use_swin
        if in_flow:
            self.stn = SpatialTransformer(size)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        # self.corr = Correlation3D(in_channel, kernel_size, d, sw, sf)
        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(2 * in_channel, in_channel, 3, 1, 1),
        #     nn.InstanceNorm3d(in_channel),
        #     nn.ReLU()
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(3 * in_channel, 3 * in_channel, 3, 1, 1),
        #     nn.Conv3d(3 * in_channel, 3 * in_channel, 3, 1, 1),
        #     nn.ReLU())
        self.resnet = ResNet(3 * in_channel)
        self.pa = PAM_Module(3 * in_channel)
        self.flow = nn.Conv3d(3 * in_channel, 3, 3, 1, 1)
        
        # init flow layer with small weights and bias
        # self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.nff = NFFBlock(3 * in_channel)
        if use_de:
            self.de = DE_Block(in_channel)
        if use_swin:
            self.swin = SwinBlock(dim=3 * window_size[0], window_size=window_size)

    def forward(self, x, y, flow=None):
        x0 = x
        if self.in_flow:
            flow = self.upsample(flow*2)
            x = self.stn(x, flow)
            if self.use_de:
                flow_f, M1 = self.de(y)
                flow_m, M2 = self.de(x)
                x1 = self.stn(M1, flow_m)
                x2 = self.stn(M2, flow_f)
                x = x + x1 + x2
            

        # stack = torch.cat([x0, x, y], dim=1)
        stack = self.nff(x0, x, y)
        
        if self.use_swin:
            stack = self.pa(stack)
        # stack = self.conv2(stack)
        stack = self.resnet(stack)
        
        if self.use_swin:
            stack = self.swin(stack)
        flow = self.flow(stack)
        if self.use_de:
            flow = flow + flow_f + flow_m
        return flow

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out
    
class NFFBlock(nn.Module):
    def __init__(self, channel):
        super(NFFBlock, self).__init__()

        c = channel
        self.conv = nn.Sequential(
            ConvInsBlock(c, c, 3, 1),
            ConvInsBlock(c, c, 3, 1)
        )
        self.weight_conv = nn.Sequential(
            nn.Conv3d(c, 3, 3, 1, 1),
            nn.Softmax(dim=1)
        )
        self.channel_attention = ChannelAttention(c)

    def forward(self, float_fm, fixed_fm, decon_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, decon_fm], dim=1)
        x = self.conv(concat_fm)
        weight_map = self.weight_conv(x)
        concat = torch.cat([
            float_fm * weight_map[:, 0, ...].unsqueeze(1),
            fixed_fm * weight_map[:, 1, ...].unsqueeze(1),
            decon_fm * weight_map[:, 2, ...].unsqueeze(1)
        ], dim=1) # (1, 3*c, h, w, t)
        channel_wise = self.channel_attention(concat)
        return concat*channel_wise

class GSAAttention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        b, c, h, w, d = x.shape
        x = x.reshape(b, -1, c)
        
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape(b, c, h, w, d)
        return x
    
class LSAAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=4):
        assert ws != 1
        super(LSAAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        b, c, h, w, d = x.shape
        x = x.reshape(b, -1, c)
        
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x.reshape(b, c, h, w, d)
        return x

class LGAttention(nn.Module):
    def __init__(self, dim, shape):
        super(LGAttention, self).__init__()
        c, h, w, d = shape
        self.ln = nn.LayerNorm(normalized_shape=(3 * c, h, w, d))
        self.ffn = nn.Sequential(nn.Conv3d(dim, 2 * dim, 3, 1, 1),
                                 nn.ReLU(),
                                 nn.Conv3d(2 * dim, dim, 3, 1, 1),
                                 nn.Dropout(0.1))
        self.lsa = LSAAttention(dim)
        self.gsa = GSAAttention(dim)
        
    def forward(self, x):
        b, c, h, w, d = x.shape
        
        # first
        x_ln = self.ln(x)
        x_gsa = self.lsa(x_ln, h, w*d)
        x_ffn = self.ffn(x_gsa)
        x = x + x_ffn
        
        x_ln = self.ln(x)
        x_lsa = self.gsa(x_ln, h, w*d)
        x_ffn = self.ffn(x_lsa)
        x = x + x_ffn
        
        # second
        x_ln = self.ln(x)
        x_gsa = self.lsa(x_ln, h*w, d)
        x_ffn = self.ffn(x_gsa)
        x = x + x_ffn
        
        x_ln = self.ln(x)
        x_lsa = self.gsa(x_ln, h*w, d)
        x_ffn = self.ffn(x_lsa)
        out = x + x_ffn
        
        return out


if __name__ == "__main__":
    x = torch.rand(1, 128, 9, 11, 9)
    print(x.shape)
    net = LGAttention(dim=128, shape=(9, 11, 9))
    y = net(x)
    print("y: ", y.shape)

    