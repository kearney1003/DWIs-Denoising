import os
from dipy.io.image import load_nifti, save_nifti
from os.path import splitext
from os import listdir
import numpy as np
import torch
from torch.utils.data import Dataset

'''
Mind that you should name your data as:

training data: train.nii.gz
ground truth: gt.nii.gz
brain mask: mask.nii.gz

According to the thesis DeepDTI:
The training data contained 9 volumes with noise: 6b1 + b0 + t1w + t2w
The validation data contained 7 volumes without noise: 6b1 + b0

If you choose to apply data normalization of your own data, the script will
load cal_norm.txt obtained by get_norm.py, or you'll use our normalization parameters,
which calculated by 100 samples of HCP dataset.
'''


class BasicDataset(Dataset):

    def __init__(self, data_dir, do_norm):
        # id
        self.ids = [splitext(file)[0] for file in listdir(data_dir)]
        self.dir = data_dir
        self.do_norm = do_norm

    def get_path(self, x):

        idx = self.ids[x]

        dwi_dir = os.path.join(self.dir, idx, 'data.nii.gz')
        gt_dir = os.path.join(self.dir, idx, 'gt.nii.gz')
        mask_dir = os.path.join(self.dir, idx, 'mask.nii.gz')

        return dwi_dir, gt_dir, mask_dir

    def __len__(self):
        return len(self.ids)

    def normalization(self, data, gt, mask):
        if self.do_norm:
            norm = np.load('cal_norm.npy', allow_pickle=True).item()
        else:
            norm = np.load('norm.npy', allow_pickle=True).item()
            
        std_train = norm['std_train']
        mean_train = norm['mean_train']
        std_gt = norm['std_gt']
        mean_gt = norm['mean_gt']

        for i in range(data.shape[3]):
            if i < gt.shape[3]: gt[:, :, :, i] = ((gt[:, :, :, i] - mean_gt[i]) / std_gt[i]) * mask
            data[:, :, :, i] = ((data[:, :, :, i] - mean_train[i]) / std_train[i]) * mask
        return data, gt

    def __getitem__(self, i):

        dwi_path, gt_path, mask_path = self.get_path(i)

        dwi = load_nifti(dwi_path, return_img=False)[0]
        gt = load_nifti(gt_path, return_img=False)[0]
        mask = load_nifti(mask_path, return_img=False)[0]

        dwi, gt = self.normalization(dwi, gt, mask)

        return torch.tensor(dwi.transpose((3, 0, 1, 2)), dtype=torch.float32), \
               torch.tensor(gt.transpose((3, 0, 1, 2)), dtype=torch.float32)
