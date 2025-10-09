import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def calculate_hd95_asd_lpba(pred, gt):
    # pred = pred.astype(np.uint8)
    # gt = gt.astype(np.uint8)
    # dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=(1.0, 1.0, 1.0))  # (2.0, 2.0, 2.0)
    asd = metric.binary.assd(pred, gt)
    return hd95, asd

def calculate_hd95_asd_ixi(pred, gt):
    # pred = pred.astype(np.uint8)
    # gt = gt.astype(np.uint8)
    # dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=(2.0, 2.0, 2.0))  # (2.0, 2.0, 2.0)
    asd = metric.binary.assd(pred, gt, voxelspacing=(2.0, 2.0, 2.0))
    return hd95, asd
def calculate_hd95_asd_l2r(pred, gt):
    # pred = pred.astype(np.uint8)
    # gt = gt.astype(np.uint8)
    # dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=(1.75, 1.25, 1.75))  # (2.0, 2.0, 2.0)
    asd = metric.binary.assd(pred, gt, voxelspacing=(1.75, 1.25, 1.75))
    return hd95, asd

def calculate_hd95_asd_ctmr(pred, gt):
    # pred = pred.astype(np.uint8)
    # gt = gt.astype(np.uint8)
    # dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt, voxelspacing=(1.0, 1.0, 1.0))  # (2.0, 2.0, 2.0)
    asd = metric.binary.assd(pred, gt)
    return hd95, asd

def calculate_nmi(volume1, volume2, bins=256):
    """
    计算两个体积图像的归一化互信息 (NMI)
    
    :param volume1: 图像1, numpy 数组
    :param volume2: 图像2, numpy 数组
    :param bins: 直方图的分箱数，默认256
    :return: 归一化互信息 (NMI) 值
    """
    # 归一化输入数据到 [0, 1]
    volume1 = (volume1 - volume1.min()) / (volume1.max() - volume1.min())
    volume2 = (volume2 - volume2.min()) / (volume2.max() - volume2.min())
    
    # 检查两个体积图像的形状是否一致
    if volume1.shape != volume2.shape:
        raise ValueError("The shapes of the two volumes must be the same.")
    
    # 计算联合直方图
    joint_hist, _, _ = np.histogram2d(volume1.ravel(), volume2.ravel(), bins=bins, range=[[0, 1], [0, 1]])
    
    # 转为概率分布
    joint_prob = joint_hist / joint_hist.sum()
    
    # 计算边缘分布
    prob1 = joint_prob.sum(axis=1)  # 图像1的边缘概率
    prob2 = joint_prob.sum(axis=0)  # 图像2的边缘概率
    
    # 计算熵
    H1 = -np.sum(prob1 * np.log(prob1 + 1e-10))  # 避免log(0)
    H2 = -np.sum(prob2 * np.log(prob2 + 1e-10))
    H_joint = -np.sum(joint_prob * np.log(joint_prob + 1e-10))
    
    # 计算归一化互信息
    nmi = (H1 + H2) / H_joint
    return nmi

def calculate_ncc(image1, image2):
    """
    计算两张三维图像的归一化互相关 (NCC) 指标。

    参数:
        image1 (numpy.ndarray): 第一张输入图像，形状为 [h, w, d]。
        image2 (numpy.ndarray): 第二张输入图像，形状为 [h, w, d]。
    
    返回:
        float: NCC 指标，范围为 [-1, 1]。
    """
    # 检查输入图像的形状是否一致
    assert image1.shape == image2.shape, "两张图像的形状必须相同"
    
    # 将图像展平为一维数组
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()
    
    # 计算均值
    mean1 = np.mean(image1_flat)
    mean2 = np.mean(image2_flat)
    
    # 计算去均值后的数组
    image1_demean = image1_flat - mean1
    image2_demean = image2_flat - mean2
    
    # 计算 NCC
    numerator = np.sum(image1_demean * image2_demean)
    denominator = np.sqrt(np.sum(image1_demean**2) * np.sum(image2_demean**2))
    ncc = numerator / denominator if denominator != 0 else 0

    return ncc

def compute_img_ncc_nmi(warpped_img, fixed_img):
    nmi = calculate_nmi(warpped_img, fixed_img)
    ncc = calculate_ncc(warpped_img, fixed_img)
    return ncc, nmi




def compute_label_dice_hd_asd_oasis(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,22,33,34,35]
    dice_lst = []
    hd_lst = []
    asd_lst = []
    i = 1
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        # print(dice)
        if dice < 0.05:
            hd = 5
            asd = 2
        else:
            hd, asd = calculate_hd95_asd_ixi(pred == cls, gt == cls)
        dice_lst.append(dice)
        hd_lst.append(hd)
        asd_lst.append(asd)
        # print("hd95: ", hd95)
        print("i: {}, dice: {}, hd95: {}, asd: {}".format(i, dice, hd, asd))
        i += 1
    return np.mean(dice_lst), np.mean(hd_lst), np.mean(asd_lst)

def compute_label_dice_hd_asd_ixi(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,28,29,30,31,22,33]
    dice_lst = []
    hd_lst = []
    asd_lst = []
    i = 1
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        # print(dice)
        if dice < 0.05:
            hd = 5
            asd = 2
        else:
            hd, asd = calculate_hd95_asd_ixi(pred == cls, gt == cls)
        dice_lst.append(dice)
        hd_lst.append(hd)
        asd_lst.append(asd)
        # print("hd95: ", hd95)
        print("i: {}, dice: {}, hd95: {}, asd: {}".format(i, dice, hd, asd))
        i += 1
    return np.mean(dice_lst), np.mean(hd_lst), np.mean(asd_lst)

def compute_label_dice_hd_asd_lpba(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 181, 182]
    dice_lst = []
    hd_lst = []
    asd_lst = []
    i = 1
    for cls in cls_lst:
        dice = DSC(gt == cls, pred == cls)
        # print(dice)
        if dice < 0.05:
            hd = 5
            asd = 2
        else:
            hd, asd = calculate_hd95_asd_lpba(pred == cls, gt == cls)
        dice_lst.append(dice)
        hd_lst.append(hd)
        asd_lst.append(asd)
        # print("hd95: ", hd95)
        print("i: {}, dice: {}, hd95: {}, asd: {}".format(i, dice, hd, asd))
        i += 1
    return np.mean(dice_lst), np.mean(hd_lst), np.mean(asd_lst)

def compute_njd(displacement):
    b, c, h, w, d = displacement.shape
    if isinstance(displacement, np.ndarray):
        displacement = displacement
    else:
        displacement = displacement.detach().cpu().numpy()
    if c==3:
        displacement = np.transpose(displacement, (0, 2, 3, 4, 1))

    D_x = (displacement[:,1:,:-1,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_y = (displacement[:,:-1,1:,:-1,:] - displacement[:,:-1,:-1,:-1,:])

    D_z = (displacement[:,:-1,:-1,1:,:] - displacement[:,:-1,:-1,:-1,:])

    D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

    D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_z[...,0])

    D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

    #D = np.abs(D1-D2+D3)

    img = D1-D2+D3
    b,c,h,d = img.shape
    n = (img<0)
    s = n.astype(int)
    njd = s.sum() / (b*c*h*d)
    return njd

# 示例用法
if __name__ == "__main__":
    # 生成两个示例体积图像
    vol1 = np.random.rand(100, 100, 100)
    vol2 = np.random.rand(100, 100, 100)
    
    # 归一化到 [0, 1]
    vol1 = (vol1 - vol1.min()) / (vol1.max() - vol1.min())
    vol2 = (vol2 - vol2.min()) / (vol2.max() - vol2.min())
    
    nmi_value = calculate_nmi(vol1, vol2)
    print("归一化互信息 (NMI):", nmi_value)
