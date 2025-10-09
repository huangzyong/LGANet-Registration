import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

'''
通过继承Data.Dataset, 实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__, __len__和__getitem__方法
'''


class Dataset_l2r(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        sample = {}
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        
        source_path = self.files[index]
        target_path = source_path.replace('0000', '0001')  # 0001 是对应的label
        sample['id'] = target_path.split('/')[-1][:11]  # LungCT_0017
        # print("id: ", sample['id'])
        # print(source_path)
        # print(target_path)
        
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(source_path))[np.newaxis, ...]
        img_arr = img_arr[:, 44:140, :, :]
        # img_arr = (img_arr - np.mean(img_arr)) / (np.std(img_arr) + 1e-4)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        
        tar_arr = sitk.GetArrayFromImage(sitk.ReadImage(target_path))[np.newaxis, ...]
        tar_arr = tar_arr[:, 44:140, :, :]  # 1 96 192 192
        # tar_arr = (tar_arr - np.mean(tar_arr)) / (np.std(tar_arr) + 1e-4)
        tar_arr = (tar_arr - tar_arr.min()) / (tar_arr.max() - tar_arr.min())
        # print(tar_arr.shape)
        # 返回值自动转换为torch的tensor类型
        sample['image'], sample['target'] = img_arr, tar_arr
        return sample