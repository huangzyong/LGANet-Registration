"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def Get_Ja(displacement):

    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''

    D_y = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_x = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])
    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])
    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])
    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    return D1-D2+D3

def NJ_loss(ypred): 
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5*(torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)

class DiceLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, moving, fixed):
        smooth = 1e-5
        pred = (moving>0.1).float()
        target = (fixed>0.1).float()
        m1 = pred.flatten()
        m2 = target.flatten()
        intersection = (m1 * m2).sum()
        return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
        

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class NCCLoss:

    def __init__(self, window=(9, 9, 9)):
        self.win = window
     
    def __call__(self, I, J, weight=False):
        sum_weight = torch.ones((1, 1, self.win[0], self.win[1], self.win[2]),
            device=I.device, dtype=torch.float)

        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute local sum
        padding = [ int((k)/2) for k in self.win ]
        I_sum = F.conv3d(I, sum_weight, padding=padding)
        J_sum = F.conv3d(J, sum_weight, padding=padding)
        I2_sum = F.conv3d(I2, sum_weight, padding=padding)
        J2_sum = F.conv3d(J2, sum_weight, padding=padding)
        IJ_sum = F.conv3d(IJ, sum_weight, padding=padding)

        # cross correlation
        win_size = self.win[0] * self.win[1] * self.win[2]
        I_u = I_sum/win_size
        J_u = J_sum/win_size

        cross = IJ_sum - J_u*I_sum - I_u*J_sum + I_u*J_u*win_size
        I_var = I2_sum - 2*I_u*I_sum + I_u*I_u*win_size
        J_var = J2_sum - 2*J_u*J_sum + J_u*J_u*win_size

        cc = cross*cross / (I_var*J_var + 1e-5)

        #weight in different slice
        if weight:
            cc = torch.mean(cc, dim=[0,1,3,4])
            w = torch.zeros(cc.size(0), dtype=cc.dtype, device = cc.device)
            for i in range(0, w.size(0), 4):
                w[i] = 1.0/w.size(0)*4.0
            cc = cc*w

        return 1.0 - cc.mean()
        # return cc


class WeightedNCCLoss:
    def __init__(self, window=(9, 9, 9), rio_size=(9, 9, 9)):
        """
        :param window: NCC计算时使用的窗口大小
        :param rio_size: 划分图像的区域数，表示每个维度上分成多少部分
        """
        self.win = window
        self.rio_size = rio_size  # 控制图像划分的区域数量
    
    def __call__(self, I, J, weight=False):
        # 计算整体NCC损失
        b,c,h,w,d = I.shape

        sum_weight = torch.ones((1, 1, self.win[0], self.win[1], self.win[2]), device=I.device, dtype=torch.float)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        # 计算局部区域的NCC
        padding = [int((k) / 2) for k in self.win]
        I_sum = F.conv3d(I, sum_weight, padding=padding)
        J_sum = F.conv3d(J, sum_weight, padding=padding)
        I2_sum = F.conv3d(I2, sum_weight, padding=padding)
        J2_sum = F.conv3d(J2, sum_weight, padding=padding)
        IJ_sum = F.conv3d(IJ, sum_weight, padding=padding)

        # 计算整体NCC
        win_size = self.win[0] * self.win[1] * self.win[2]
        I_u = I_sum / win_size
        J_u = J_sum / win_size
        cross = IJ_sum - J_u * I_sum - I_u * J_sum + I_u * J_u * win_size
        I_var = I2_sum - 2 * I_u * I_sum + I_u * I_u * win_size
        J_var = J2_sum - 2 * J_u * J_sum + J_u * J_u * win_size
        cc = cross * cross / (I_var * J_var + 1e-5)

        # 计算局部区域相似性
        region_size = (I.shape[2] // self.rio_size[0], 
                       I.shape[3] // self.rio_size[1], 
                       I.shape[4] // self.rio_size[2])  # 划分后每个区域的大小
        region_weights = []

        for z in range(self.rio_size[0]):
            for y in range(self.rio_size[1]):
                for x in range(self.rio_size[2]):
                    # 计算每个局部区域的NCC相似性
                    z_start = z * region_size[0]
                    z_end = (z + 1) * region_size[0]
                    y_start = y * region_size[1]
                    y_end = (y + 1) * region_size[1]
                    x_start = x * region_size[2]
                    x_end = (x + 1) * region_size[2]

                    # 获取局部区域
                    # local_cc = cc[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                    # local_cc = (I[:, :, z_start:z_end, y_start:y_end, x_start:x_end] - J[:, :, z_start:z_end, y_start:y_end, x_start:x_end])**2

                    tensor1 = I[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                    tensor2 = J[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                    tensor1_flat = tensor1.reshape(b, c, -1)  # [b, c, N]
                    tensor2_flat = tensor2.reshape(b, c, -1)  # [b, c, N]
                    
                    # 计算余弦相似度
                    similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=2)  # [b, c]
                    
                    # 对通道求平均，得到每个样本的余弦相似度
                    similarity = similarity.mean()  # [b]

                    region_weights.append(similarity)
        # x = I[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
        # print(region_weights.sshape)
        # print(len(region_weights))
        # 归一化每个局部区域的相似性到 [1, 10]
        min_weight = min(region_weights)
        max_weight = max(region_weights)

        region_weights = [(w - min_weight) / (max_weight - min_weight) for w in region_weights]

        # 创建加权NCC的新tensor
        weighted_cc = cc.clone()

        # 遍历局部区域并加权
        i = 0
        for z in range(self.rio_size[0]):
            for y in range(self.rio_size[1]):
                for x in range(self.rio_size[2]):
                    # 计算每个局部区域的NCC相似性
                    z_start = z * region_size[0]
                    z_end = (z + 1) * region_size[0]
                    y_start = y * region_size[1]
                    y_end = (y + 1) * region_size[1]
                    x_start = x * region_size[2]
                    x_end = (x + 1) * region_size[2]

                    # 获取局部区域
                    local_cc = cc[:, :, z_start:z_end, y_start:y_end, x_start:x_end]
                    # 将加权后的区域存入weighted_cc中
                    weighted_cc[:, :, z_start:z_end, y_start:y_end, x_start:x_end] = (1 - region_weights[i]) * local_cc
                    i += 1

        # 计算加权NCC损失
        # weighted_ncc_loss = 1 - weighted_cc.mean()
        weighted_ncc_loss = torch.exp(5*(-weighted_cc.mean()))

        return weighted_ncc_loss


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = torch.tensor(temperature).cuda()  # 超参数 温度
        self.negatives_mask = torch.eye(batch_size * 2, batch_size * 2, dtype=bool).float().cuda()
        
    def forward(self, emb_i, emb_j): # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到 (bs, c, h, w, d)
        b, c, h, w, d = emb_i.size()
        emb_i = emb_i.view(b, -1)
        emb_j = emb_j.view(b, -1)
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2) # simi_mat: (2*bs, 2*bs)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class LocalLoss(nn.Module):
    def __init__(self, max_num=3, neighborhood_size=3, **kwargs):
        super().__init__()
        self.max_num = max_num
        self.neighborhood_size = neighborhood_size

        
    # 获取每个最大值的 8 邻域 (3x3x3)
    def get_neighborhood(self, tensor, center, neighborhood_size=3):
        """从 tensor 中获取中心点的 3x3x3 邻域, 并处理边界"""
        # 计算邻域的范围
        half_size = neighborhood_size // 2
        z_start, z_end = max(center[0] - half_size, 0), min(center[0] + half_size + 1, tensor.shape[0])
        if z_end==tensor.shape[0]:
            z_start = z_end-neighborhood_size
        if z_start==0:
            z_end=z_start+neighborhood_size
        y_start, y_end = max(center[1] - half_size, 0), min(center[1] + half_size + 1, tensor.shape[1])
        if y_end==tensor.shape[1]:
            y_start = y_end-neighborhood_size
        if y_start==0:
            y_end=y_start+neighborhood_size
        x_start, x_end = max(center[2] - half_size, 0), min(center[2] + half_size + 1, tensor.shape[2])
        if x_end==tensor.shape[2]:
            x_start = x_end-neighborhood_size
        if x_start==0:
            x_end=x_start+neighborhood_size

        # 返回该邻域的 tensor
        return tensor[z_start:z_end, y_start:y_end, x_start:x_end]
        
    def forward(self, warped_img, fixed_img, alpha):
        # 配准之后，计算 warped_img 和 fixed_img 之间的损失，同时找出局部损失比较大的地方，单独计算这一局部位置的损失，用于更新网络。
        mse = (fixed_img-warped_img)**2
        
        a,b,c,d,e = mse.shape

        # 将五维张量展平为一维张量，方便找到最大5个值
        flattened_tensor = mse.flatten()
        values, indices = torch.topk(flattened_tensor.view(a, -1), self.max_num)

        
        # 对每个 batch 的 top 5 位置获取 3x3x3 邻域 tensor
        neighborhood_fixed = torch.empty(mse.size(0), self.max_num, self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        neighborhood_warped = torch.empty(mse.size(0), self.max_num, self.neighborhood_size, self.neighborhood_size, self.neighborhood_size)
        for batch_idx in range(mse.size(0)):
            for i in range(self.max_num):
                
                idx = indices[batch_idx, i]
                z = idx // (b * c * d * e)  # 第一维索引
                remainder = idx % (b * c * d * e)
                y = remainder // (c * d * e)  # 第二维索引
                remainder = remainder % (c * d * e)
                x = remainder // (d * e)      # 第三维索引
                remainder = remainder % (d * e)
                w = remainder // e             # 第四维索引
                v = remainder % e               # 第五维索引
                
                
                center = (x, w, v) # 取每个最大值的中心坐标
                # print("center: ", center)
                # print(mse[batch_idx, 0, center[0], center[1], center[2]])
                neighborhood_f = self.get_neighborhood(fixed_img[batch_idx, 0], center, self.neighborhood_size)
                neighborhood_w = self.get_neighborhood(warped_img[batch_idx, 0], center, self.neighborhood_size)
                neighborhood_fixed[batch_idx, i, ...] = neighborhood_f
                neighborhood_warped[batch_idx, i, ...] = neighborhood_w
        mse_local = (neighborhood_fixed-neighborhood_warped)**2
        print("mse: {}, local loss: {}".format(mse.mean(), mse_local.mean()))
        return mse.mean() + 0.5 * alpha * mse_local.mean()
        # return mse.mean()


class GISRegLoss(torch.nn.Module):
    def __init__(self, sim_loss='MSE', reduction='mean', window_size=9, eps=1e-5):
        super(GISRegLoss, self).__init__()
        self.sim_loss = sim_loss
        self.reduction = reduction
        self.window_size = window_size
        self.eps = eps
        self.lncc = NCCLoss(window=(9,9,9))

    def compute_gradient(self, img):
        ndims = len(img.shape) - 2
        if ndims == 2:
            grad_x = img[..., 1:, :] - img[..., :-1, :]
            grad_x = F.pad(grad_x, (0, 0, 0, 1), mode='replicate')
            grad_y = img[..., :, 1:] - img[..., :, :-1]
            grad_y = F.pad(grad_y, (0, 1, 0, 0), mode='replicate')
            grad_z = None
        elif ndims == 3:
            grad_x = img[..., 1:, :, :] - img[..., :-1, :, :]
            grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1), mode='replicate')
            grad_y = img[..., :, 1:, :] - img[..., :, :-1, :]
            grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0), mode='replicate')
            grad_z = img[..., :, :, 1:] - img[..., :, :, :-1]
            grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0), mode='replicate')
        else:
            raise ValueError(f"Unsupported spatial dimensions: {ndims}")
        return grad_x, grad_y, grad_z

    def lncc_loss(self, I, J):
        ndims = len(I.shape) - 2
        window = [self.window_size] * ndims
        sum_filter = torch.ones([1, 1, *window], device=I.device)
        if ndims == 3:
            pad_size = [self.window_size//2]*6
            conv_fn = F.conv3d
        elif ndims == 2:
            pad_size = [self.window_size//2]*4
            conv_fn = F.conv2d
        else:
            raise ValueError(f"Unsupported spatial dims: {ndims}")

        I2, J2, IJ = I*I, J*J, I*J

        I_sum = conv_fn(I, sum_filter, padding=pad_size)
        J_sum = conv_fn(J, sum_filter, padding=pad_size)
        I2_sum = conv_fn(I2, sum_filter, padding=pad_size)
        J2_sum = conv_fn(J2, sum_filter, padding=pad_size)
        IJ_sum = conv_fn(IJ, sum_filter, padding=pad_size)

        window_size = torch.prod(torch.tensor(window, device=I.device))
        u_I, u_J = I_sum / window_size, J_sum / window_size
        cross = IJ_sum - u_I * u_J * window_size
        I_var = I2_sum - u_I * u_I * window_size
        J_var = J2_sum - u_J * u_J * window_size
        cc = cross * cross / (I_var * J_var + self.eps)
        loss = 1 - cc
        return loss.mean() if self.reduction == 'mean' else loss.sum()

    def _warp_gradient(self, grad, phi):
        b, c, d, h, w = grad.shape
        device = grad.device

        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.arange(d, device=device),
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )

        grid = torch.stack((grid_w, grid_h, grid_d), dim=-1).float()
        grid = grid.unsqueeze(0).repeat(b,1,1,1,1)
        phi_perm = phi.permute(0, 2, 3, 4, 1)
        new_grid = grid + phi_perm

        new_grid[..., 0] = 2.0 * new_grid[..., 0] / (w - 1) - 1
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / (h - 1) - 1
        new_grid[..., 2] = 2.0 * new_grid[..., 2] / (d - 1) - 1

        warped = F.grid_sample(grad, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    def forward(self, M_warped, Fixed):
        grad_Fixed = self.compute_gradient(Fixed)
        grad_M_warped = self.compute_gradient(M_warped)

        loss = 0
        count = 0
        for gFixed, gMw in zip(grad_Fixed, grad_M_warped):
            if gFixed is None or gMw is None:
                continue
            if self.sim_loss == 'MSE':
                loss += F.mse_loss(gFixed, gMw, reduction=self.reduction)
            elif self.sim_loss == 'LNCC':
                loss += self.lncc_loss(gFixed, gMw)
            count += 1
        if count == 0:
            return torch.tensor(0., device=Fixed.device)
        # print("count: ", count)
        return loss




if __name__ == "__main__":
    x = torch.rand(1,1,144,176,144)
    y = torch.rand(1,1,144,176,144)
    phi = torch.rand(1,3,144,176,144)
    cre = GISRegLoss()
    loss = cre(x, y, phi)
    print("loss: ", loss)
    # print("loss: ", s.shape)
