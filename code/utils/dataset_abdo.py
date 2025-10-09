import os
import glob
import itertools
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data

from random import randint

'''
通过继承Data.Dataset, 实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__, __len__和__getitem__方法
'''


class Dataset_abdo(Data.Dataset):
    def __init__(self, files, mode='train'):
        # 初始化
        self.files = files
        self.mode = mode
        # if mode=='train':
        #     self.index_pair = list(itertools.permutations(files, 2))

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        sample = {}
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        
        source_path = self.files[index]
        MR_path = source_path
        # print("source_path: ", source_path)
        # print("###"*5)
        if self.mode == 'train':
            ct_str = source_path.split("_")[1][:4]
            num = int(ct_str)
            # print("num: ", num)
            if num < 6:
                MR_path = source_path.replace('0001.', '0000.')  # 0000 是对应的fixed img
            elif (num > 1000) and (num < 1051):
                index_num=randint(1051-num, 1090-num)
                num_mr = num+index_num
                # print("num_mr: ", num_mr)
                mr_str = str(num_mr)
                # print("mr_str: ", type(mr_str))
                MR_path = source_path.replace(ct_str, mr_str)  # MR 是对应的fixed img
                MR_path = MR_path.replace('0001.', '0000.') 
                # print("MR_path: ", MR_path)
            # source_path = self.index_pair[index][0]
            # MR_path = self.index_pair[1][0]
            
            

        else:
            ct_str = source_path.split("_")[1][:4]
            num = int(ct_str)
            if (num < 9):
                MR_path = source_path.replace('0001.', '0000.')  # 0000 是对应的fixed img
            
        sample['id'] = source_path.split('/')[-1][:16]  # LungCT_0017
        # print("id: ", sample['id'])
        # print(source_path)
        # print(MR_path)
        
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(source_path))[np.newaxis, ...]
        img_arr = img_arr[:, 24:168, :, :]
        # img_arr = (img_arr - np.mean(img_arr)) / (np.std(img_arr) + 1e-4)
        img_arr = np.clip(img_arr, -800,800)
        # print("min: ", img_arr.min(), "max: ", img_arr.max())
        # img_arr = (img_arr - np.mean(img_arr)) / (np.std(img_arr) + 1e-4)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        
        tar_arr = sitk.GetArrayFromImage(sitk.ReadImage(MR_path))[np.newaxis, ...]
        tar_arr = tar_arr[:, 24:168, :, :]  # 1 112 192 192   39:151  32:160
        tar_arr = (tar_arr - np.mean(tar_arr)) / (np.std(tar_arr) + 1e-4)
        # print("min: ", tar_arr.min(), "max: ", tar_arr.max())
        tar_arr = (tar_arr - tar_arr.min()) / (tar_arr.max() - tar_arr.min())
        # 返回值自动转换为torch的tensor类型
        sample['image'], sample['target'] = img_arr, tar_arr
        return sample