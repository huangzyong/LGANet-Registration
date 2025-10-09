import sys
import glob
import datetime
import shutil
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import warnings
import pandas as pd
import SimpleITK as sitk
import argparse
from sklearn.model_selection import KFold
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from predict_lpba40 import compute_label_dice
# import ex_transforms
from natsort import natsorted
from tensorboardX import SummaryWriter
from utils.dataset import Dataset
from utils.util import get_run_name, get_scheduler, update_learning_rate
from models import losses
from models.model import U_Network, SpatialTransformer, Dual_Unet

from models.LGANetPlusPlus import LGANetPlusPlus
from models.LGANet import LGANet



warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1

curr_time = datetime.datetime.now()

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument("--modelType", type=str, help="voxelmorph 1 or 2", choices=['vm1', 'vm2'], default='vm2')
parser.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                    dest="sim_loss", default='ncc')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=250, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of num_works for data loader to use')
parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')

parser.add_argument('--model', type=str, default='LGANet', help='CorrMLP | VM, PRNetplusplus, ModeT, Pivit, PCNet')
parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
parser.add_argument('--gpu', type=int, default=1, help='select gpu')
opt = parser.parse_args()

def save_image(img, ref_img, name, save_result_folder):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    save_result_folder = os.path.join(save_result_folder, 'save_img')
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)
    sitk.WriteImage(img, os.path.join(save_result_folder, name))


if __name__ == '__main__':
    curr_time = datetime.datetime.now()
    Path_trainset = r'../data/LPBA40/train'
    Path_fixed_img = r'../data/LPBA40/fixed.nii.gz' 
    
    
    run_name = get_run_name()
    date = run_name.split('_')[0]
    save_result_folder = '../Results/LPBA40/' + opt.model +"/"+ opt.model + "_" + date
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)

     # make logger file
    if os.path.exists(save_result_folder + '/code'):
        shutil.rmtree(save_result_folder + '/code')
    shutil.copytree('.', save_result_folder + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=save_result_folder + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(opt))

    writer = SummaryWriter(save_result_folder+'/log')

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
    
    # ********* TODO Select CUDA Device *********
    device = torch.device("cuda:{}".format(opt.gpu)) 
    print(device)
    

    print('===> Loading datasets')
    # 读入fixed图像
    f_img = sitk.ReadImage(Path_fixed_img)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    input_fixed = input_fixed[:, :, 8:152, 8:184, 8:152]
    input_fixed_image = torch.from_numpy(input_fixed).to(device).float()
    vol_size = input_fixed.shape[2:]
    # [B, C, D, W, H]
    input_fixed = np.repeat(input_fixed, opt.batch_size, axis=0)
    input_fixed_img = torch.from_numpy(input_fixed).to(device).float()
    
    train_files = []
    file_list = os.listdir(Path_trainset)
    for f in file_list:
        file = os.path.join(Path_trainset, f)
        train_files.append(file)
    DS = Dataset(files=train_files)
    print("Number of training images: ", len(DS))
    train_set_loader = DataLoader(DS, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=True)

    print("train data: {}".format(len(train_set_loader)*opt.batch_size))

    print('===> Building model')
    
    # validation
    label_dir = '/home/hzy/Projects/MICCAI-25/data/LPBA40/label'
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_dir, "S01.delineation.structure.label.nii.gz")))
    fixed_label = fixed_label[8:152, 8:184, 8:152]
    test_dir = '/home/hzy/Projects/MICCAI-25/data/LPBA40/test'
    test_file_lst = glob.glob(os.path.join(test_dir, "*.nii.gz"))
    
    # 创建配准网络（UNet）和STN
    nf_enc = [16, 32, 32, 32]
    if opt.modelType == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    
    if opt.model == 'VM':
        net = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    elif opt.model == 'LGANet++':
        net = LGANetPlusPlus(vol_size).to(device)
    elif opt.model == 'LGANet':
        net = LGANet(vol_size).to(device)
    
    else:
        raise ValueError("input net error")
    
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)  # 计算模型参数量
    total_params_mb = total_params / (1024 * 1024)  # 将参数转换为MB
    logging.info(f"Total Parameters: {total_params} ({total_params_mb:.3f} MB)")


    STN = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)

        
    criterion_ncc_loss = losses.NCCLoss(window=(9,9,9)) 

    criterion_grad_loss = losses.gradient_loss


    criterionL1 = nn.L1Loss().to(device)

    # set optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.99, 0.999), weight_decay=0.0005)
    # optimizer = optim.AdamW(net.parameters(), lr=opt.lr, betas=(0.99, 0.999), weight_decay=0.005)
    net_scheduler = get_scheduler(optimizer, opt)
    
    all_epoch = opt.niter + opt.niter_decay 
    
    mean_DSC = 0
    best_epoch = 0
    for epoch in range(opt.epoch_count, all_epoch + 1):
        # train
        net.train()
        STN.train()
   
        for iteration, sample in enumerate(train_set_loader, 1):
            
            # print('batch: ', batch.shape)
            image, id = sample['image'], sample['id']
            input_moving_img = image.to(device).float()

            flow_img = net(input_moving_img, input_fixed_img)

            m2f = STN(input_moving_img, flow_img)
            # print(m2f.shape)
            # print(fixed_img.shape)
 

            optimizer.zero_grad()

            sim_loss = criterion_ncc_loss(m2f, input_fixed_img)
            grad_loss = criterion_grad_loss(flow_img)

            
            loss = sim_loss + grad_loss
            # loss = wncc_loss + grad_loss
            loss.backward()
            optimizer.step()
            if iteration % 10==0:
                logging.info("Epoch[{}/{}]({}/{}): Loss: {:.4f}, Sim_loss: {:.4f} Grad_loss: {:.4f}".format(
                    epoch, all_epoch, iteration, len(train_set_loader), loss.item(), sim_loss.item(), grad_loss.item()))
        
        update_learning_rate(net_scheduler, optimizer)
        # update_learning_rate(net_scheduler_d, optimizer_d)
        lr = optimizer.param_groups[0]['lr']
        logging.info("LR: {}".format(lr))
        
        writer.add_scalar('epoch_loss', loss, epoch + 1)
        writer.add_scalar('ncc_loss', sim_loss, epoch + 1)

        writer.add_scalar('grad_loss', grad_loss, epoch + 1)
        writer.add_scalar('lr', lr, epoch + 1)
        
        del loss, sim_loss, grad_loss, m2f, flow_img
        
        # validation
        net.eval()
        STN.eval()
        STN_label.eval()
        DSC = []
        for file in test_file_lst:
            name = os.path.split(file)[1].split('.')[0]
            # 读入moving图像
            input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
            input_moving = input_moving[:, :, 8:152, 8:184, 8:152]
            input_moving = torch.from_numpy(input_moving).to(device).float()
            # 读入moving图像对应的label
            label_file = glob.glob(os.path.join(label_dir, name[:3] + "*"))[0]
            input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
            input_label = input_label[:, :, 8:152, 8:184, 8:152]
            input_label = torch.from_numpy(input_label).to(device).float()
            # print('input_moving: ', input_moving.shape)
            # print('input_fixed_image: ', input_fixed_image.shape)
            # print('input_label: ', input_label.shape)

            # 获得配准后的图像和label
            # print("input_fixed_image: ", input_fixed_image.shape,)
            flow_img = net(input_moving, input_fixed_image)
            # print('pred_flow: ', pred_flow.shape)
            pred_img = STN(input_moving, flow_img)
            pred_label = STN_label(input_label, flow_img)

            # 计算DSC
            dice = compute_label_dice(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
            logging.info("id: {}, dice: {:.8f}".format(name, dice))
            DSC.append(dice)
            del dice, flow_img, pred_img, pred_label
        mean_dice = np.mean(DSC)
        writer.add_scalar('mean_dsc', mean_dice, epoch + 1)
        print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))
        logging.info("mean_dice: {:.8f} mean_std: {:.8f}".format(np.mean(DSC), np.std(DSC)))
        
        if mean_DSC < mean_dice:
            best_epoch = epoch
            mean_DSC = mean_dice
            save_best_path = os.path.join(save_result_folder, "save_model")
            if not os.path.exists(save_best_path):
                os.makedirs(save_best_path)
            save_model_name = os.path.join(save_best_path, "model_best.pth")
            torch.save({'net': net.state_dict(), 'epoch': epoch}, save_model_name)
            logging.info("save model to {}, epoch is {}".format(save_best_path, epoch))
        logging.info("Best_mean_dice: {:.8f} epoch: {}".format(mean_DSC, best_epoch))
           
        
        if epoch % opt.save_freq == 0:
            save_model_path = os.path.join(save_result_folder, "save_model")
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)
            save_model_name = os.path.join(save_model_path, str(epoch) + ".pth")
            torch.save({'net': net.state_dict(), 'epoch': epoch}, save_model_name)
            logging.info("save model to {}, epoch is {}".format(save_model_path, epoch))
     
    curr_time_1 = datetime.datetime.now()
    print("计算时间:", curr_time_1 - curr_time)



