import torch
import nibabel
import monai
import numpy as np
from matplotlib import pyplot as plt
import random
import SimpleITK as sitk
import math
import re
import cv2

def dose_to_cond(dose_value, unit="mAs", min_val=None, max_val=None):
    """
    将原始剂量值转换为条件输入（纯 Python/NumPy 版本）：
    1. 统一单位到 mAs
    2. log 压缩
    3. Min-Max 归一化到 [0,1]

    参数:
        dose_value: float 或可转为 float
        unit: "mAs" 或 "uAs"
        min_val: 数据集 log 压缩后的最小值
        max_val: 数据集 log 压缩后的最大值

    返回:
        cond: float，归一化后的条件值
    """

    # 转换到 mAs
    if unit == "uAs":
        dose_mAs = float(dose_value) / 1000.0
    else:
        dose_mAs = float(dose_value)

    # log 压缩
    dose_log = np.log1p(dose_mAs)

    # 如果没有提供 min/max，则使用自身（单样本时返回 0）
    if min_val is None:
        min_val = dose_log
    if max_val is None:
        max_val = dose_log

    # 归一化
    cond = (dose_log - min_val) / (max_val - min_val + 1e-8)

    return float(cond)

class CFM_train(torch.utils.data.Dataset):
    def __init__(self, list_IDs, name=None):
        self.list_IDs = list_IDs
        self.name = name
        self.clip_min = 0
        self.clip_max = 300

    def __len__(self):
        return len(self.list_IDs['hepatic'])

    def __getitem__(self, index):
        ct = nibabel.load(self.list_IDs['ct'][index]).get_fdata().transpose(2,0,1)
        # filtered_ct = nibabel.load(self.list_IDs['filtered_ct'][index]).get_fdata().transpose(2,0,1)
        hepatic = nibabel.load(self.list_IDs['hepatic'][index]).get_fdata().transpose(2,0,1)
        portal = nibabel.load(self.list_IDs['portal'][index]).get_fdata().transpose(2,0,1)

        ct = np.clip(ct, self.clip_min, self.clip_max)
        ct = (ct-self.clip_min) / (self.clip_max-self.clip_min+1e-7)
        # ct = (ct-ct.min()) / (ct.max()-ct.min()+1e-7)
        # filtered_ct = (filtered_ct-filtered_ct.min()) / (filtered_ct.max()-filtered_ct.min()+1e-7)

        label = hepatic + portal
        label[label>0] = 1

        # ct_25d = torch.cat((torch.from_numpy(ct)[None].type(torch.float32), torch.from_numpy(filtered_ct)[None].type(torch.float32)), dim=1)
        ct_25d = torch.from_numpy(ct)[None].type(torch.float32)
        # ct = ct_25d[:,3,].unsqueeze(1)
        label = torch.from_numpy(label)[None].type(torch.LongTensor)
        label = label[:,3,].unsqueeze(1)

        return tuple((ct_25d, label))