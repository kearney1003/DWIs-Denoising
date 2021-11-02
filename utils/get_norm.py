from dipy.io.image import load_nifti, save_nifti
import os
import glob
import numpy as np

'''
get_norm(train_dir, val_dir)

train_dir: the folder you save the training samples
val_dir: the folder you save the validation samples

In this script, we provide a function to help you obtain the normalization statistics.

The script will walk through the train directory and validation directory and get all the samples.

The statistics is volume-wised, which means the train_mean returns an array with the shape of [1*9] 
for those 9 volumes designed in DeepDTI, the same to others

The script returns a dictionary, the keys are:

mean_train
mean_gt
std_train
std_gt

********************************* WARNING *********************************
Mind we collect the training samples by glob function using glob.glob(*data*),
if your database did not follow this format, you need to revise the code manually,
or revise your filenames (does not recommend)

:)  

'''

def get_norm(train_dir, val_dir):
    # initial
    mean_train = np.zeros(9)
    mean_gt = np.zeros(7)
    std_train = np.zeros(9)
    std_gt = np.zeros(7)
    norm = dict()
    file_list = []

    train_list = os.listdir(train_dir)
    for fn in train_list:
        file_list.append(os.path.join(train_dir, fn))

    val_list = os.listdir(val_dir)
    for fn in val_list:
        file_list.append(os.path.join(val_dir, fn))

    file_length = 0
    for fn in file_list:

        # get all the subset
        trainset = glob.glob(os.path.join(fn, '*data*'))
        gtset = glob.glob(os.path.join(fn, '*gt*'))

        file_length += len(trainset)

        for train in trainset:
            data = load_nifti(os.path.join(train))[0]
            for i in range(9):
                mean_train[i] += np.mean(data[:, :, :, i])
                std_train[i] += np.std(data[:, :, :, i])

        for gt in gtset:
            gt_data = load_nifti(os.path.join(gt))[0]
            for j in range(7):
                mean_gt[j] += np.mean(gt_data[:, :, :, j])
                std_gt[j] += np.std(gt_data[:, :, :, j])

    norm['mean_train'] = mean_train / file_length
    norm['mean_gt'] = mean_gt / file_length
    norm['std_train'] = std_train / file_length
    norm['std_gt'] = std_gt / file_length

    # save the normalization parameters in a dictionary
    np.save(os.path.join('cal_norm.npy'), norm)
    return norm
