import torch
import os
import numpy as np
from dipy.io.image import load_nifti, save_nifti
from CNN.network import DeepCNN
import math
from skimage.metrics import structural_similarity
import argparse
from utils.get_norm import get_norm

'''
A test script, aiming to evaluate the model using several approaches,
and all the test scores are obtained separately for each volume.
1. MSE
2. PSNR
3. SSIM
4. Error map (residual, optional)
The test scores will show on the terminal,

Please arrange the data in following path:
|_DWIs-Denoising
    |_test
        |_'sample name'
            |_data.nii.gz
            |_gt.nii.gz
            |_mask.nii.gz
    norm.npy
    cal_norm.npy

Or you can name the test directory in command line by:
python test.py --test_dir [test_dir]

The data inside still needs to be arranged as:
|_test_dir
    |_'sample name'
        |_data.nii.gz
        |_gt.nii.gz
        |_mask.nii.gz

'''

# Compare MSE, SSIM between the original noised DWIs and denoised DWIs. 
def evaluate(data, gt, pred):
    mse_noised, psnr_noised = get_psnr(data, gt)
    mse_denoised, psnr_denoised = get_psnr(pred, gt)

    ssim_noised = get_ssim(data, gt)
    ssim_denoised = get_ssim(pred, gt)

    print('The ORIGINAL MSE, PSNR and SSIM for 7 volumes are:\n')
    print('MSE (Billion):', [m / 1e9 for m in mse_noised])
    print('Mean MSE (Billion):', sum(mse_noised) / len(mse_noised) / 1e9, '\n')
    print('PSNR (dB):', psnr_noised)
    print('Mean PSNR (dB):', sum(psnr_noised) / len(psnr_noised), '\n')
    print('SSIM:', ssim_noised)
    print('Mean SSIM:', sum(ssim_noised) / len(ssim_noised), '\n')

    print('***********************************************')

    print('The DENOISED MSE and PSNR for 7 volumes are:\n')
    print('MSE (Billion):', [m / 1e9 for m in mse_denoised])
    print('Mean MSE (Billion):', sum(mse_denoised) / len(mse_denoised) / 1e9, '\n')
    print('PSNR (dB):', psnr_denoised)
    print('Mean PSNR (dB):', sum(psnr_denoised) / len(psnr_denoised), '\n')
    print('SSIM:', ssim_denoised)
    print('Mean SSIM:', sum(ssim_denoised) / len(ssim_denoised), '\n')


# Get predicted volumes using our well-trained model
def test(x, model, mask):

    #   Normalization parameters
    std_gt = norm_dict['std_gt']
    mean_gt = norm_dict['mean_gt']
    std_train = norm_dict['std_train']
    mean_train = norm_dict['mean_train']
    x = normalization(x, mean_train, std_train, mask)

    x = torch.tensor(x.transpose((3, 0, 1, 2))).unsqueeze(0)  # input shape (1, 9, 145, 174, 145)
    y = model(x)
    y = y.squeeze(0).detach().numpy().transpose((1, 2, 3, 0))  # output shape (1, 7, 145 ,175, 145)

    y = denormalization(y, mean_gt, std_gt, mask)

    # return a numpy array, the shape is (145, 174, 145, 7)
    return y


# Volume-wised Z-score normalization and denormalization function
def normalization(volume, mean_, std_, mask):
    for i in range(volume.shape[3]):
        volume[:, :, :, i] = ((volume[:, :, :, i] - mean_[i]) / std_[i]) * mask
    return volume


def denormalization(volume, mean_, std_, mask):
    for i in range(volume.shape[3]):
        volume[:, :, :, i] = (volume[:, :, :, i] * std_[i] + mean_[i]) * mask
    return volume


# Get MSE and PSNR to evaluate the denoised effect
def get_psnr(data, gt):
    psnr_list = []
    mse_list = []
    for volume in range(gt.shape[3]):
        mse = 0
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                for k in range(gt.shape[2]):
                    mse += (gt[i, j, k, volume] - data[i, j, k, volume]) ** 2
        mse_list.append(mse)
        n = data.shape[0] * data.shape[1] * data.shape[2]
        psnr_list.append(10 * math.log10((gt[:, :, :, volume].max()) ** 2 / (mse / n)))
    return mse_list, psnr_list


# Get SSIM to measure the structural similarity
def get_ssim(data, gt):
    ssim_list = []
    for volume in range(gt.shape[3]):
        score = structural_similarity(data[:, :, :, volume], gt[:, :, :, volume], data_range=1.0)
        ssim_list.append(score)
    return ssim_list


# Get the error map
def get_errormap(data, gt):
    error_map = abs(gt - data)
    return error_map


def config_argparser(p):
    parser.add_argument('--name', help='The name of subject', type=str)
    parser.add_argument('--no_error', help='Plot error map (before and after) or not', action='store_false')
    parser.add_argument('--test_dir', help='Name the test directory, the default is in test/',
                        default='test/', type=str)
    parser.add_argument('--no_norm', help='Normalization the data or not', action='store_false')
    parser.add_argument('--cal_norm', help='Use the norm calculated statistics', action='store_false')
    parser.add_argument('--model', help='Name the path of test model', type=str, default='model.pth')
    return p


def main():
    mask = load_nifti(os.path.join('test', sn, 'mask.nii.gz'))[0]

    #    Load weights of the network
    net = DeepCNN(mid_channels=24)
    net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    print('The model has been loaded!\n')

    #       Load data
    data_dir = os.path.join('test', sn, 'data.nii.gz')
    gt_dir = os.path.join('test', sn, 'gt.nii.gz')
    data, affine = load_nifti(data_dir)
    gt = load_nifti(gt_dir)[0]

    #       Get predicted volumes
    pred = test(data, net, mask)
    save_nifti(os.path.join(pred_dir, 'prediction.nii.gz'), pred, affine)
    print('The prediction  has been saved!\n')

    #       Evaluate the predicted output
    data, affine = load_nifti(data_dir)
    evaluate(data, gt, pred)

    if plot_error:
        #   Draw error maps
        error_noised = get_errormap(data[:, :, :, :7], gt)
        error_denoised = get_errormap(pred, gt)
        save_nifti(os.path.join(pred_dir, 'error_noised.nii.gz'), error_noised, affine)
        print('The noised error map has been saved!\n')
        save_nifti(os.path.join(pred_dir, 'error_denoised.nii.gz'), error_denoised, affine)
        print('The denosied error map has been saved!\n')


if __name__ == "__main__":
    #   Config the argparser
    parser = argparse.ArgumentParser()
    parser = config_argparser(parser)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1 '

    sn = args.name  # sample name
    test_dir = args.test_dir  # output directory
    plot_error = args.no_error  # whether plot error map
    do_norm = args.no_norm  # whether do normalization
    cal_norm = args.cal_norm  # whether use calculated statistics
    model_dir = args.model  # The directory of the test model

    if test_dir == 'test/' and not os.path.exists(test_dir):
        os.mkdir(test_dir)

    if not os.path.exists(model_dir):
        print("Wrong path of the named model")
        os._exit(1)

    if do_norm:
        if cal_norm:
            norm_dict = np.load('norm.npy', allow_pickle=True).item()
        else:
            norm_dict = get_norm(test_dir, test_dir)

    pred_dir = os.path.join(test_dir, sn)

    main()
