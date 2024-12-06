import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)

def initialize_weights_kaiming(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.InstanceNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
# def initialize_weights_kaiming(m):
#     if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
#         nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias.data, 0.0)
#     elif isinstance(m, nn.InstanceNorm3d):
#         nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
#         nn.init.constant_(m.bias.data, 0.0)

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

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, size, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(size)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        x = x[:,:,1:-1,1:-1,1:-1]
        return self.actout(x)

class DeconvBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels):
        super(DeconvBlock, self).__init__()
        self.upconv = UpConvBlock(dec_channels, skip_channels)
        self.conv = nn.Sequential(
            ConvInsBlock(2*skip_channels, skip_channels),
            ConvInsBlock(skip_channels, skip_channels)
        )
    def forward(self, dec, skip):
        dec = self.upconv(dec)
        out = self.conv(torch.cat([dec, skip], dim=1))
        return out

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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=3):
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
        x_gsa = self.lsa(x_ln, h, w*d)
        x_ffn = self.ffn(x_gsa)
        x = x + x_ffn
        
        x_ln = self.ln(x)
        x_lsa = self.gsa(x_ln, h*w, d)
        x_ffn = self.ffn(x_lsa)
        out = x + x_ffn
        
        return out

    
class LSAAttention3D(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, ws=(6,6,6)):  # 6 375 for oasis
        assert ws != 1
        super(LSAAttention3D, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = x.reshape(b, -1, c)
        
        B, N, C = x.shape
        h_group, w_group, d_group = h // self.ws[0], w // self.ws[1], d //self.ws[2]

        total_groups = h_group * w_group * d_group

        x = x.reshape(B, total_groups, -1, C)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v)
        x = self.proj(attn)
        x = self.proj_drop(x)  # B T N C
        x = x.permute(0, 3, 1, 2)  # B C T N
        x = x.reshape(b, c, h, w, d)
        return x

class GSAAttention3D(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Linear(dim, dim//8),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(dim//8, dim))
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w, d = x.shape
        x = x.reshape(b, -1, c)
        
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, h, w, d)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            k = self.k(x)
            v = self.v(x)
        else:
            k = self.k(x)
            v = self.v(x)
        
        # print("v: ", v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B N N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # print("attn: ", attn.shape)
        # print("attn @ v: ", (attn @ v).shape)
        x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.transpose(1, 2)  # B C N
        x = x.reshape(b, c, h, w, d)
        return x

class LGAttention3D(nn.Module):
    def __init__(self, dim):
        super(LGAttention3D, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Conv3d(dim, 4 * dim, 1),
                                 nn.ReLU(),
                                 nn.Conv3d(4 * dim, dim, 1),
                                 nn.Dropout(0.1))
        self.lsa = GSAAttention3D(dim)
        self.gsa = LSAAttention3D(dim)
        
    def forward(self, x):
        b, c, h, w, d = x.shape
        
        # first
        x1 = x.permute(0, 2, 3, 4, 1)
        x_ln = self.ln(x1).permute(0, 4, 1, 2, 3)
        x_gsa = self.lsa(x_ln)
        x_ffn = self.ffn(x_gsa)
        x = x + x_ffn
        
        x1 = x.permute(0, 2, 3, 4, 1)
        x_ln = self.ln(x1).permute(0, 4, 1, 2, 3)
        x_lsa = self.gsa(x_ln)
        x_ffn = self.ffn(x_lsa)
        x = x + x_ffn
        
        # second
        x1 = x.permute(0, 2, 3, 4, 1)
        x_ln = self.ln(x1).permute(0, 4, 1, 2, 3)
        x_gsa = self.lsa(x_ln)
        x_ffn = self.ffn(x_gsa)
        x = x + x_ffn
        
        x1 = x.permute(0, 2, 3, 4, 1)
        x_ln = self.ln(x1).permute(0, 4, 1, 2, 3)
        x_lsa = self.gsa(x_ln)
        x_ffn = self.ffn(x_lsa)
        out = x + x_ffn
        
        return out


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

class ResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, padding=1),
            # nn.Conv3d(channel, channel, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        res = self.block(x)
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



class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=8):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channel, c, 3, 1, 1),
            ConvInsBlock(c, c),
            ConvInsBlock(c, c),
            # ResidualBlockShift(c, c)
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2 * c, 3, 1, 1),
            ConvInsBlock(2 * c, 2 * c),
            ConvInsBlock(2 * c, 2 * c),
            # ResidualBlockShift(2 * c, 2 * c)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2*c, 4*c, 3, 1, 1),
            ConvInsBlock(4 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c),
            # ResidualBlockShift(4 * c, 4 * c)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(4*c, 8*c, 3, 1, 1),
            ConvInsBlock(8 * c, 8* c),
            ConvInsBlock(8 * c, 8 * c),
            # ResidualBlockShift(8 * c, 8 * c)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(8*c, 16*c, 3, 1, 1),
            ConvInsBlock(16 * c, 16 * c),
            ConvInsBlock(16 * c, 16 * c),
            # ResidualBlockShift(16 * c, 16 * c)
        )
        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)
        self.pool3 = nn.MaxPool3d(2)
        

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(self.pool0(out0))  # 1/2
        out2 = self.conv2(self.pool1(out1))  # 1/4
        out3 = self.conv3(self.pool2(out2))  # 1/8
        out4 = self.conv4(self.pool3(out3))  # 1/16

        return out0, out1, out2, out3, out4

class LGAMBlock(nn.Module):
    def __init__(self, dim=16, depth=1, num_heads=1, window_size=(128, 9, 11, 9)):
        super(LGAMBlock, self).__init__()
        self.attn = LGAttention(dim=dim, shape=window_size)

    def forward(self, x):
        x = self.attn(x)
        return x
     
class LGAMBlock3D(nn.Module):
    def __init__(self, dim=16, window_size=(128, 9, 11, 9)):
        super(LGAMBlock3D, self).__init__()
        self.attn = LGAttention3D(dim=dim)

    def forward(self, x):
        x = self.attn(x)
        return x

class ResBlock1(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, alpha=0.1):
        super(ResBlock1, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            nn.Conv3d(channel, channel, kernel_size=3, padding=1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)



class CFBlock(nn.Module):
    def __init__(self, channel):
        super(CFBlock, self).__init__()

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



class ResDe(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResDe, self).__init__()
        self.conv1 = ConvInsBlock(in_c, in_c)
        self.res1 = ResBlock(in_c)
        
        self.conv2 = ConvInsBlock(in_c, 2*in_c)
        self.res2 = ResBlock(2*in_c)
        
        self.conv3 = ConvInsBlock(2*in_c, 4*in_c)
        self.res3 = ResBlock(4*in_c)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x) + x
        
        x = self.conv2(x)
        x = self.res2(x) + x
        
        x = self.conv3(x)
        x = self.res3(x) + x
        return x
                
class FIFM_Block(nn.Module):
    def __init__(self, in_c):
        super(FIFM_Block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, 2 * in_c, 3, 1, 1),
            nn.InstanceNorm3d(2*in_c),
            nn.Conv3d(2 * in_c, 2 * in_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(2 * in_c, 4 * in_c, 3, 1, 1),
            nn.InstanceNorm3d(2*in_c),
            nn.Conv3d(4 * in_c, 4 * in_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(4 * in_c, 4 * in_c, 3, 1, 1),
            nn.Conv3d(4 * in_c, 4 * in_c, 3, 1, 1),
            nn.ReLU(inplace=True))
        # self.conv = ResDe(in_c=in_c, out_c=4*in_c)
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(4 * in_c, 2 * in_c, 3, 1, 1),
            nn.Conv3d(2 * in_c, in_c, 1, 1, 0),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(4 * in_c, 2 * in_c, 3, 1, 1),
            nn.Conv3d(2 * in_c, 3, 1, 1, 0),
            nn.ReLU())
        
    def forward(self, x):
        s = self.conv(x)
        M = self.conv1(s)
        flow = self.conv2(s)
        return flow, M

class Decoder(nn.Module):
    def __init__(self, size, in_channel, in_flow=True, use_da=False, use_de=False, use_lgam=False, window_size=(128, 9, 11, 9)):
        super(Decoder, self).__init__()
        self.in_flow = in_flow
        self.use_da = use_da
        self.use_de = use_de
        self.use_lgam = use_lgam
        if in_flow:
            self.stn = SpatialTransformer(size)
            self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.resnet = ResNet(3 * in_channel)
        self.resnets = ResNet(2 * in_channel)
        self.pa = PAM_Module(2 * in_channel)
 
        self.flow = nn.Conv3d(3 * in_channel, 3, 3, 1, 1)
        self.flows = nn.Conv3d(2 * in_channel, 3, 3, 1, 1)
        
        self.cf = CFBlock(3 * in_channel)
        if use_de:
            self.de = FIFM_Block(in_channel)
        if use_lgam:
            self.swin = nn.Sequential(LGAMBlock3D(dim=2 * in_channel, window_size=window_size),
                                      LGAMBlock3D(dim=2 * in_channel, window_size=window_size)
                                      )
            
        self.convs = ConvInsBlock(2 * in_channel, 2 * in_channel)

      
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

        if self.use_lgam:
            stack = self.convs(torch.cat([x, y], dim=1))
            stack = self.pa(stack)
            stack = self.resnets(stack)
            stack = self.swin(stack)
            flow = self.flows(stack)
        else:
            stack = self.cf(x0, x, y)
            stack = self.resnet(stack)
            flow = self.flow(stack)
            if self.use_de:
                flow = flow + flow_f + flow_m
        
        return flow

class LGANet(nn.Module):
    def __init__(self,
                 size=(144,176,144),
                 in_channel=1,
                 filters=8):
        super(LGANet, self).__init__()
        self.channels = filters
        self.size = size

        self.encoder = Encoder(in_channel=in_channel, first_out_channel=filters)
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)  # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
 

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        
        self.deblock5 = Decoder(size=[s // 16 for s in self.size], in_channel=16*filters, in_flow=False, use_de=False, use_lgam=True, window_size=(16*filters, 6, 12, 12))
        self.deblock4 = Decoder(size=[s // 8 for s in self.size], in_channel=8*filters, in_flow=True, use_de=True)
        self.deblock3 = Decoder(size=[s // 4 for s in self.size], in_channel=4*filters, in_flow=True, use_de=True)
        self.deblock2 = Decoder(size=[s // 2 for s in self.size], in_channel=2*filters, in_flow=True, use_de=True)
        self.deblock1 = Decoder(size=[s for s in self.size], in_channel=filters, in_flow=True, use_de=False)
        
        # self.apply(initialize_weights_kaiming)
        self.integrate5 = VecInt(size=[s // 16 for s in self.size])
        self.integrate4 = VecInt(size=[s // 8 for s in self.size])
        self.integrate3 = VecInt(size=[s // 4 for s in self.size])
        self.integrate2 = VecInt(size=[s // 2 for s in self.size])
        self.integrate1 = VecInt(size=size)
        
    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)  # 8 16 32 64 128
        F1, F2, F3, F4, F5 = self.encoder(fixed)
      
        flow5 = self.deblock5(M5, F5)  # 1 3 9 11 9
        flow5 = self.integrate5(flow5)
        
        flow4 = self.deblock4(M4, F4, flow5)  # 1 3 18 22 18
        flow4 = self.integrate4(flow4)
        flow4 = self.transformer[3](self.upsample_trilin(2*flow5), flow4)+flow4
        
        
        flow3 = self.deblock3(M3, F3, flow4)  # 1 3 36 44 36
        flow3 = self.integrate3(flow3)
        flow3 = self.transformer[2](self.upsample_trilin(2*flow4), flow3)+flow3
        
        
        flow2 = self.deblock2(M2, F2, flow3)  # 1 3 72 88 72
        flow2 = self.integrate2(flow2)
        flow2 = self.transformer[1](self.upsample_trilin(2*flow3), flow2)+flow2
        
        
        flow = self.deblock1(M1, F1, flow2)  # 1 3 144 176 144
        flow = self.integrate1(flow)
        flow = self.transformer[0](self.upsample_trilin(2*flow2), flow)+flow

        
        return flow

if __name__ == '__main__':
    input = torch.rand(1, 1, 144, 176, 144)  # 6 12 12  12 24 24   24 48 48  48 96 96  96 192 192
    model = LGANet()
    flow = model(input, input)
    print(flow.shape)
