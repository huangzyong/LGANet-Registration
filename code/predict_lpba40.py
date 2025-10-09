# python imports
import os
import sys
import glob
import logging
import argparse
import datetime
# external imports
import torch
import numpy as np
import SimpleITK as sitk
# internal imports
from models import losses, metrics
from utils.util import get_run_name
from models.model import U_Network, SpatialTransformer, Dual_Unet

from models.LGANetPlusPlus import LGANetPlusPlus
from models.LGANet import LGANet
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# def make_dirs():
#     if not os.path.exists(args.result_dir):
#         os.makedirs(args.result_dir)


def save_image(img, ref_img, folder, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(folder, name))

def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 181, 182]
    dice_lst = []

    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
      
        dice_lst.append(dice)
    return np.mean(dice_lst)

def compute_label_dice_hd_asd(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 181, 182]
    dice_lst = []
    hd_lst = []
    asd_lst = []
    i=1
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        hd, asd = metrics.calculate_hd95_asd_lpba(pred == cls, gt == cls)
        dice_lst.append(dice)
        hd_lst.append(hd)
        asd_lst.append(asd)
        print("cls: {}, dice: {}, hd95: {}, asd: {}".format(cls, dice, hd, asd))
        i += 1
    return np.mean(dice_lst), np.mean(hd_lst), np.mean(asd_lst)



def compute_acc_f1(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166, 181, 182]

    acc_lst, f1_lst, precision_lst, recall_lst = [], [], [], []

    for cls in cls_lst:
        gt_mask = (gt == cls).astype(np.uint8).flatten()
        pred_mask = (pred == cls).astype(np.uint8).flatten()

        # Accuracy & F1
        acc = accuracy_score(gt_mask, pred_mask)
        precision = precision_score(gt_mask, pred_mask, zero_division=0)
        recall = recall_score(gt_mask, pred_mask, zero_division=0)
        f1 = f1_score(gt_mask, pred_mask, zero_division=0)  # 防止分母为0报错
        acc_lst.append(acc)
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        print(f"cls: {cls}, acc: {acc:.4f}, f1: {f1:.4f}")

    return (np.mean(acc_lst), np.mean(recall_lst), np.mean(precision_lst), np.mean(f1_lst))


# @torchsnooper.snoop()
def test():
    run_name = get_run_name()
    date = run_name.split('_')[0]
    # make_dirs()
    device = torch.device("cuda:{}".format(args.gpu))
    print(args.checkpoint_path)

    result_dir = os.path.join(args.checkpoint_path.split('save_model')[0],"test_result_acc_f1_lganet")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    logging.basicConfig(filename=result_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Test Time: {}".format(date))

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    input_fixed = input_fixed[:, :, 8:152, 8:184, 8:152]
    vol_size = input_fixed.shape[2:]
    fixed_image = input_fixed[0, 0, ...]
    # set up atlas tensor
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    model = args.checkpoint_path.split('/')[-4]
    print(model)

    if model=='VM':
        UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)

    elif model == 'LGANet++':
        UNet = LGANetPlusPlus(vol_size).to(device)
    elif model == 'LGANet':
        UNet = LGANet(vol_size).to(device)
    else:
        raise ValueError("input net error")
    UNet.load_state_dict(torch.load(args.checkpoint_path)['net'])
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()
    DSC = []
    HD = []
    ASD = []
    Acc = []
    Recall = []
    Precision = []
    F1 = []
    # fixed图像对应的label
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))
    fixed_label = fixed_label[8:152, 8:184, 8:152]
    
    fixed_save = torch.from_numpy(fixed_label[np.newaxis, np.newaxis, ...])
    
    for file in test_file_lst:
        name = os.path.split(file)[1]
        # 读入moving图像
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
        input_moving = input_moving[:, :, 8:152, 8:184, 8:152]
        moving_image = input_moving[0, 0, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        
        # 读入moving图像对应的label
        label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
        input_label = input_label[:, :, 8:152, 8:184, 8:152]
        input_label = torch.from_numpy(input_label).to(device).float()

        # 获得配准后的图像和label
        s_time = datetime.datetime.now()
        print(s_time)
        pred_flow = UNet(input_moving, input_fixed)
        pred_img = STN_img(input_moving, pred_flow)
        pred_label = STN_label(input_label, pred_flow)
        # print("pred_label: ", pred_label.shape)
        # print("fixed_label: ", fixed_label.shape)
        e_time = datetime.datetime.now()
        print(e_time)
        print("time: ", e_time - s_time)

        # 计算 DSC, HD, ASSD
        print("computing...")
        # dice, hd, asd = compute_label_dice_hd_asd(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
        dice, hd, asd = 0, 0, 0
        print("dice: ", dice)
        print("hd: ", hd)
        print("asd: ", asd)
        DSC.append(dice)
        HD.append(hd)
        ASD.append(asd)
        # 计算 NCC, NMI
        acc, recall, precision, f1 = compute_acc_f1(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
        Acc.append(acc)
        Recall.append(recall)
        Precision.append(precision)
        F1.append(f1)
        logging.info("name: {:} dice: {:.8f} hd: {:.8f} asd: {:.8f} acc: {:.8f} recall: {:.8f}, precision: {:.8f} f1: {:.8f}".format(name[:3], dice, hd, asd, acc, recall, precision, f1)) 

        # save image
        folder = os.path.join(result_dir, "save_image", name[:3])
        if not os.path.exists(folder):
            os.makedirs(folder)
        # file_name = name[:3]
        # save_image(input_moving, f_img, folder, "moving.nii.gz")
        # save_image(input_label, f_img, folder, "moving_label.nii.gz")
        # save_image(pred_img, f_img, folder, "warped.nii.gz")
        # save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, folder, "flow.nii.gz")
        # save_image(pred_label, f_img, folder, "pred_label.nii.gz")
        
        # save_image(input_fixed, f_img, folder, "fixed.nii.gz")
        # save_image(fixed_save, f_img, folder, "fixed_label.nii.gz")
        
        del pred_flow, pred_img, pred_label

    # print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC)) 
    # print("mean(HD): ", np.mean(HD), "   std(HD): ", np.std(HD))    
    # print("mean(ASD): ", np.mean(ASD), "   std(ASD): ", np.std(ASD)) 
    # print("mean(NCC): ", np.mean(NCC), "   std(HD): ", np.std(NCC))    
    # print("mean(NMI): ", np.mean(NMI), "   std(ASD): ", np.std(NMI)) 
    logging.info("Average score:")
    logging.info("mean_dice: {:.8f} mean_std: {:.8f}".format(np.mean(DSC), np.std(DSC)))  
    logging.info("mean_hd: {:.8f} mean_std: {:.8f}".format(np.mean(HD), np.std(HD))) 
    logging.info("mean_asd: {:.8f} mean_std: {:.8f}".format(np.mean(ASD), np.std(ASD))) 
    logging.info("mean_acc: {:.8f} mean_std: {:.8f}".format(np.mean(Acc), np.std(Acc))) 
    logging.info("mean_recall: {:.8f} mean_std: {:.8f}".format(np.mean(Recall), np.std(Recall))) 
    logging.info("mean_precision: {:.8f} mean_std: {:.8f}".format(np.mean(Precision), np.std(Precision))) 
    logging.info("mean_f1: {:.8f} mean_std: {:.8f}".format(np.mean(F1), np.std(F1))) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # 公共参数
    parser.add_argument("--gpu", type=str, help="gpu id",
                        dest="gpu", default='1')
    parser.add_argument("--atlas_file", type=str, help="gpu id number",
                        dest="atlas_file", default='../data/LPBA40/fixed.nii.gz')
    parser.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                        dest="model", choices=['vm1', 'vm2'], default='vm2')
    # parser.add_argument("--result_dir", type=str, help="results folder",
    #                 dest="result_dir", default='../Results_LPBA40_Test_on_OASIS/VM')
    parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='../data/LPBA40/test')
    parser.add_argument("--label_dir", type=str, help="label data directory",
                        dest="label_dir", default='../data/LPBA40/label')
    parser.add_argument("--checkpoint_path", type=str, help="model weight file",
                    dest="checkpoint_path", default="../Results/LPBA40/LGANet/LGANet_Mar09-02-54-36/save_model/model_300.pth")
    args = parser.parse_args()
    test()
