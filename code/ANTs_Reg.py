import os
import sys
import glob
import ants
import numpy as np
import SimpleITK as sitk
import logging
from models import metrics
from utils.util import get_run_name
import predict_lpba40



def ANTs_Reg(fixed_path, fixed_label_path, moving_path, moving_label_path, output_path, name, dataset):
    # ants图片的读取
    # f_img = ants.image_read("/home/hzy/Projects/MICCAI-25/data/LPBA40/fixed.nii.gz")
    # m_img = ants.image_read("/home/hzy/Projects/MICCAI-25/data/LPBA40/train/S11.delineation.skullstripped.nii.gz")
    # f_label = ants.image_read("/home/hzy/Projects/MICCAI-25/data/LPBA40/label/S01.delineation.structure.label.nii.gz")
    # m_label = ants.image_read("/home/hzy/Projects/MICCAI-25/data/LPBA40/label/S11.delineation.structure.label.nii.gz")

    print("begin!")
    f_img = ants.image_read(fixed_path)
    f_label = ants.image_read(fixed_label_path)
    m_img = ants.image_read(moving_path)
    m_label = ants.image_read(moving_label_path)

    print("begin reg !")
    # 图像配准
    mytx = ants.registration(fixed=f_img, moving=m_img, type_of_transform='SyN')
    # 将形变场作用于moving图像，得到配准后的图像，interpolator也可以选择"nearestNeighbor"等
    warped_img = ants.apply_transforms(fixed=f_img, moving=m_img, transformlist=mytx['fwdtransforms'],
                                    interpolator="linear")
    # 对moving图像对应的label图进行配准
    warped_label = ants.apply_transforms(fixed=f_img, moving=m_label, transformlist=mytx['fwdtransforms'],
                                        interpolator="nearestNeighbor")
    # 将配准后图像的direction/origin/spacing和原图保持一致
    warped_img.set_direction(f_img.direction)
    warped_img.set_origin(f_img.origin)
    warped_img.set_spacing(f_img.spacing)
    warped_label.set_direction(f_img.direction)
    warped_label.set_origin(f_img.origin)
    warped_label.set_spacing(f_img.spacing)
    print("end!")
    # img_name = "/home/hzy/Projects/MICCAI-25/Result_ANTs/warped_img.nii.gz"
    # label_name = "/home/hzy/Projects/MICCAI-25/Result_ANTs/warped_label.nii.gz"
    output_name_path = os.path.join(output_path, name)
    if not os.path.exists(output_name_path):
        os.makedirs(output_name_path)
    img_name = os.path.join(output_name_path,  'warped_img.nii.gz')
    label_name = os.path.join(output_name_path, 'warped_label.nii.gz')

    # 图像的保存
    ants.image_write(warped_img, img_name)
    ants.image_write(warped_label, label_name)

    # 读取形变场文件
    def_field_path = mytx['fwdtransforms'][0]  # 获取形变场路径
    def_field_img = ants.image_read(def_field_path)  # 读取形变场

    def_field_np = def_field_img.numpy()  # 转换为 NumPy 数组 [1,h,w,d,3]
    print("形变场数组形状:", def_field_np.shape)  # 形状通常为 (H, W, D, 3)
    # 创建新的 ANTs 图像
    def_field_nii = ants.from_numpy(def_field_np, origin=def_field_img.origin, spacing=def_field_img.spacing, direction=def_field_img.direction, has_components=True)

    # 保存为 NIfTI 文件
    flow_name = os.path.join(output_name_path,  'flow.nii.gz')
    ants.image_write(def_field_nii, flow_name)
    print(f"形变场已保存至 {output_path}")



    # 生成图像的雅克比行列式
    jac = ants.create_jacobian_determinant_image(domain_image=f_img, tx=mytx["fwdtransforms"][0], do_log=False, geom=False)
    ants.image_write(jac, os.path.join(output_name_path, "jac_img.nii.gz"))

    # 将antsimage转化为numpy数组
    warped_img_arr = warped_img.numpy(single_components=False)
    warped_label_arr = warped_label.numpy(single_components=False)
    f_img = f_img.numpy(single_components=False)
    f_label = f_label.numpy(single_components=False)
    # 计算定量指标
    print("computing dice, hd, assd, ncc, nmi...")
    # if dataset == "LPBA":
    #     dice, hd, asd = metrics.compute_label_dice_hd_asd_lpba(f_label, warped_label_arr)
    # elif dataset == "OASIS":
    #     dice, hd, asd = metrics.compute_label_dice_hd_asd_oasis(f_label, warped_label_arr)
    # elif dataset == "IXI":
    #     dice, hd, asd = metrics.compute_label_dice_hd_asd_ixi(f_label, warped_label_arr)
    # else:
    #     raise ValueError("dataset error {}".format(dataset))

    dice, hd, asd = 0, 0, 0

    # ncc, nmi = predict_lpba40.compute_acc_f1(f_label, warped_label_arr)
    if dataset == "LPBA":
        acc, recall, precision, f1 = predict_lpba40.compute_acc_f1(f_label, warped_label_arr)
    elif dataset=='IXI':
        acc, recall, precision, f1 = predict_ixi.compute_acc_f1(f_label, warped_label_arr)

    def_field_np = def_field_img.numpy()[np.newaxis, ...]  # 转换为 NumPy 数组 [1,h,w,d,3]
    njd = metrics.compute_njd(def_field_np)



    return dice, hd, asd, njd, acc, recall, precision, f1

    

    # # 从numpy数组得到antsimage
    # img = ants.from_numpy(warped_img_arr, origin=None, spacing=None, direction=None, has_components=False, is_rgb=False)
    # # 生成带网格的moving图像，实测效果不好
    # m_grid = ants.create_warped_grid(m_img)
    # m_grid = ants.create_warped_grid(m_grid, grid_directions=(False, False), transform=mytx['fwdtransforms'],
    #                                 fixed_reference_image=f_img)
    # ants.image_write(m_grid, "/home/hzy/Projects/MICCAI-25/Result_ANTs/m_grid.nii.gz")

    '''
    以下为其他不常用的函数：

    ANTsTransform.apply_to_image(image, reference=None, interpolation='linear')
    ants.read_transform(filename, dimension=2, precision='float')
    # transform的格式是".mat"
    ants.write_transform(transform, filename)
    # field是ANTsImage类型
    ants.transform_from_displacement_field(field)
    '''
    # print("End")

if __name__ == '__main__':

    # LPBA
    # run_name = get_run_name()
    # date = run_name.split('_')[0]
    # test_dir = '/home/hzy/Projects/MICCAI-25/data/LPBA40/test'
    # label_dir = '/home/hzy/Projects/MICCAI-25/data/LPBA40/label'
    # fixed_path = '/home/hzy/Projects/MICCAI-25/data/LPBA40/fixed.nii.gz'
    # fixed_label_path = '/home/hzy/Projects/MICCAI-25/data/LPBA40/label/S01.delineation.structure.label.nii.gz'
    # output_path = '../Results-ANTs-re/LPBA40'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # logging.basicConfig(filename=output_path + "/log_acc_f1.txt", level=logging.INFO,
    #                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # logging.info("Test Time: {}".format(date))
 
    # test_file_lst = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir)]

    # DSC = []
    # HD = []
    # ASD = [] 
    # Acc = []
    # Recall = []
    # Precision = []
    # F1 = []
    # NJD = []

    # num = 1
    # output_path = os.path.join(output_path, 'save_img')
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # for file in test_file_lst:  # 138 for OASIS; 115 for IXI
    #     name = os.path.split(file)[1]
    #     moving_path = file
    #     moving_label_path = glob.glob(os.path.join(label_dir, name[:3]+"*.nii.gz"))[0]
    #     print("moving_label_path, ", moving_label_path)
        
    #     name = name[:3]  
    #     print("name: ", name)
    #     dice, hd, asd, njd, acc, recall, precision, f1 = ANTs_Reg(fixed_path, fixed_label_path, moving_path, moving_label_path, output_path, name, dataset="LPBA")  # LPBA, OASIS, IXI
    #     DSC.append(dice)
    #     HD.append(hd)
    #     ASD.append(asd)
    #     NJD.append(njd)
    #     Acc.append(acc)
    #     Recall.append(recall)
    #     Precision.append(precision)
    #     F1.append(f1)
    #     logging.info("num: {}, name: {} dice: {:.8f} hd95: {:.8f} assd: {:.8f} njd: {:.8f} acc: {:.8f} recall: {:.8f} precision: {:.8f} f1: {:.8f}".format(num, name, dice, hd, asd, njd, acc, recall, precision, f1)) 
    #     num += 1

    # logging.info("Average score:")
    # logging.info("mean_dice: {:.8f} mean_std: {:.8f}".format(np.mean(DSC), np.std(DSC)))  
    # logging.info("mean_hd95: {:.8f} mean_std: {:.8f}".format(np.mean(HD), np.std(HD))) 
    # logging.info("mean_assd: {:.8f} mean_std: {:.8f}".format(np.mean(ASD), np.std(ASD))) 
    # logging.info("mean_njd: {:.8f} mean_std: {:.8f}".format(np.mean(NJD), np.std(NJD))) 
    # logging.info("mean_acc: {:.8f} std_ncc: {:.8f}".format(np.mean(Acc), np.std(Acc))) 
    # logging.info("mean_recall: {:.8f} std_recall: {:.8f}".format(np.mean(Recall), np.std(Recall))) 
    # logging.info("mean_precision: {:.8f} std_pre: {:.8f}".format(np.mean(Precision), np.std(Precision))) 
    # logging.info("mean_f1: {:.8f} std_nmi: {:.8f}".format(np.mean(F1), np.std(F1))) 


    # IXI
    run_name = get_run_name()
    date = run_name.split('_')[0]
    test_dir = '/home/hzy/Projects/MICCAI-25/data/IXI-400-LPI/test'
    label_dir = '/home/hzy/Projects/MICCAI-25/data/IXI-400-LPI/test'
    fixed_path = '/home/hzy/Projects/MICCAI-25/data/IXI-400-LPI/fixed-IXI002-Guys-0828/aligned_norm_LPI.nii.gz'
    fixed_label_path = '/home/hzy/Projects/MICCAI-25/data/IXI-400-LPI/fixed-IXI002-Guys-0828/aligned_seg35_LPI.nii.gz'
    output_path = '../Results-ANTs-re/IXI-400-LPI-30-labels/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(filename=output_path + "/log_acc_f1.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Test Time: {}".format(date))


    test_file_lst = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir)]
    DSC = []
    HD = []
    ASD = [] 
    Acc = []
    Recall = []
    Precision = []
    F1 = []
    NJD = []

    num = 1
    output_path = os.path.join(output_path, 'save_img')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in test_file_lst[:115]:  # 115 for IXI
        name = os.path.split(file)[1]
        moving_path = os.path.join(test_dir, name, 'aligned_norm_LPI.nii.gz')
        moving_label_path = os.path.join(test_dir, name, 'aligned_seg35_LPI.nii.gz')
        
        # print("file:", file)
        # print("name: ", name)
        dice, hd, asd, njd, acc, recall, precision, f1 = ANTs_Reg(fixed_path, fixed_label_path, moving_path, moving_label_path, output_path, name, dataset="IXI")  # LPBA, OASIS, IXI
        DSC.append(dice)
        HD.append(hd)
        ASD.append(asd)
        NJD.append(njd)
        Acc.append(acc)
        Recall.append(recall)
        Precision.append(precision)
        F1.append(f1)
        logging.info("num: {}, name: {} dice: {:.8f} hd95: {:.8f} assd: {:.8f} njd: {:.8f} acc: {:.8f} recall: {:.8f} precision: {:.8f} f1: {:.8f}".format(num, name, dice, hd, asd, njd, acc, recall, precision, f1)) 
        num += 1

    logging.info("Average score:")
    logging.info("mean_dice: {:.8f} mean_std: {:.8f}".format(np.mean(DSC), np.std(DSC)))  
    logging.info("mean_hd95: {:.8f} mean_std: {:.8f}".format(np.mean(HD), np.std(HD))) 
    logging.info("mean_assd: {:.8f} mean_std: {:.8f}".format(np.mean(ASD), np.std(ASD))) 
    logging.info("mean_njd: {:.8f} mean_std: {:.8f}".format(np.mean(NJD), np.std(NJD))) 
    logging.info("mean_acc: {:.8f} std_ncc: {:.8f}".format(np.mean(Acc), np.std(Acc))) 
    logging.info("mean_recall: {:.8f} std_recall: {:.8f}".format(np.mean(Recall), np.std(Recall))) 
    logging.info("mean_precision: {:.8f} std_pre: {:.8f}".format(np.mean(Precision), np.std(Precision))) 
    logging.info("mean_f1: {:.8f} std_nmi: {:.8f}".format(np.mean(F1), np.std(F1))) 



    