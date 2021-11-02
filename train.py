import logging
from torch import optim
from tqdm import tqdm
from CNN.network import DeepCNN
from utils.dataset import BasicDataset
from utils.plot import plot_curve
from utils.get_norm import get_norm
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from torch import nn
import math
import argparse

'''
Define the training approach by following parameters:

net: the network;
train_dir: the directory for train samples;
val_dir: the directory for validate samples;
device: the device for calculation, cuda or cpu;
epochs: the number of training iterations;
batches: the number of batch size;
learning_rate: the learning rate;
save_cp: whether save the checkpoint after each iteration;
save_loss: whether save the loss results and loss curve;
do_init: whether initialize the network in the beginning;
do_norm: whether apply z-score norm calculated from your own dataset;

Use terminal to change the parameters, see detailed usage by the command:
python train.py -h

You are required to provide the train_dir/val_dir, and the training files needed to be renamed as train.nii.gz
the ground truth files needed to be renamed as gt.nii.gz, so does the brain mask to be mask.nii.gz

Or you can revise the code in utils.dataset.py and utils.get_norm.py by your accustomed format.

You can download our well-trained model from the drive, see link in readme.

The loss function is torch.nn.MSE;
The optimizer is torch.optim.Adam()

Sadly to say, the training process consumes the GPU memory a lot, so I bet you will encounter the MMO, congrats!
'''


def train_net(net,  # model
              train_dir,  # training directory
              val_dir,  # validation directory
              device,  # mapping device
              epochs,  # number of iterations
              batches,  # batch size
              learning_rate,  # learning rate
              save_cp,  # whether save the checkpoint
              save_loss,  # whether save the loss curves
              do_init,  # whether initialize the network
              do_norm):  # whether normalize the data
    train_dataset = BasicDataset(train_dir, do_norm)
    val_dataset = BasicDataset(val_dir, do_norm)
    logging.info(f'Creating training dataset with {len(train_dataset)} examples')
    logging.info(f'and validation dataset with {len(val_dataset)} examples')

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-8)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batches}
        Learning rate:   {learning_rate}
        Loss function:   {str(criterion)}
        Checkpoints:     {save_cp}
        Loss Curve:      {save_loss}      
        Initialization:  {do_init}
        Calculate Norms: {do_norm}
        Training dir:    {train_dir}
        Validation dir:  {val_dir}
        device:          {str(device)}
     
    ''')

    if do_init:
        net.apply(weight_init)
        logging.info('Initializing network...')

    # load data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batches, shuffle=True,
                                               pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batches, shuffle=False,
                                             pin_memory=False)

    # train epoch loss, validation epoch loss
    train_loss = []
    validate_loss = []

    # iteration loop
    for epoch in range(epochs):
        epoch_trainloss, epoch_valloss = 0, 0

        # set progress bar, start iteration
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='dwi') as pbar:

            # training
            net.train()
            for batch, (img, gt) in enumerate(train_loader):

                img = img.to(device=device, dtype=torch.float32)  # to device

                gt = gt.to(device=device, dtype=torch.float32)

                img = net(img)
                loss = criterion(img, gt)
                pbar.set_postfix(**{'loss (dwi)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update progress bar
                pbar.update(img.shape[0])
                logging.info('training batch loss:{:.3e}'.format(loss.item()))
                epoch_trainloss += loss.item()

        epoch_trainloss = epoch_trainloss / (batch+1)
        logging.info('traning epoch loss:{:.3e}'.format(epoch_trainloss))

        train_loss.append(epoch_trainloss)
        logging.info('Validating...')

        net.eval()
        with torch.no_grad():
            for val_batch, (val_img, val_gt) in enumerate(val_loader):
                val_img = val_img.to(device=device, dtype=torch.float32)
                val_gt = val_gt.to(device=device, dtype=torch.float32)
                val_img = net(val_img)

                val_loss = criterion(val_img, val_gt)

                epoch_valloss += val_loss.item()

        epoch_valloss = epoch_valloss / (val_batch+1)

        logging.info('validation epoch loss:{:.3e}'.format(epoch_valloss))
        validate_loss.append(epoch_valloss)

        if save_cp:
            torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    if save_loss:
        plot_curve(train_loss, validate_loss)
        np.savetxt(dir_results + 'train_loss.txt', train_loss)
        np.savetxt(dir_results + 'val_loss.txt', validate_loss)
        logging.info('Loss curve saved!')


def weight_init(layer):  # Initialization
    if isinstance(layer, nn.Conv3d):
        n = layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2] * layer.out_channels
        layer.weight.data.normal_(0, math.sqrt(2.0 / n))
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm3d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()


def config_argparser(p):
    p.add_argument('--train', help='Name the training directory', type=str, default='train/')
    p.add_argument('--val', help='Name the validation directory', type=str, default='val/')

    p.add_argument('-b', '--batch', help='Number the batch size, the default is 1', type=int, default=1)
    p.add_argument('-e', '--epoch', help='Number the iterations, the default is 100', type=int, default=100)
    p.add_argument('-l', '--lr', help='Change the learning rate, the default is 1e-4', type=int, default=1e-4)

    p.add_argument('--device', help='Name the device for calculation, the default is CUDA', default='cuda', type=str)
    p.add_argument('--no_cp', help='Save the checkpoint or not', action='store_false')
    p.add_argument('--no_curve', help='Save the loss curve or not', action='store_false')
    p.add_argument('--no_init', help='Initial the model or not', action='store_false')
    p.add_argument('--cal_norm', help='Normalize the data with the parameters obtained by your own dataset or not', action='store_false')
    return p


def exception_eval(arg):
    train_dir = arg.train
    if not os.path.exists(train_dir):
        logging.info('ERROR! The training directory does not exist, please input the right directory.')
        os._exit(1)

    val_dir = arg.val
    if not os.path.exists(val_dir):
        logging.info('ERROR! The validation directory does not exist, please input the right directory.')
        os._exit(1)

    dev = arg.device
    if dev == 'cuda' and (not torch.cuda.is_available()):
        logging.info('ERROR! The CUDA is not available in current environment, '
                     'please check your cuda version and related configuration')
        os._exit(1)

    save_cp = arg.no_cp
    if save_cp and (not os.path.exists(dir_checkpoint)):
        os.mkdir(dir_checkpoint)

    save_curve = arg.no_curve
    if save_curve and (not os.path.exists(dir_results)):
        os.mkdir(dir_results)

    norm = arg.cal_norm
    if norm:
        norm_dict = get_norm(train_dir, val_dir)
        logging.info('Created a dictionary contained statistics for z-score normalization using your own dataset')


if __name__ == '__main__':
    # Config parser
    parser = argparse.ArgumentParser()
    parser = config_argparser(parser)
    args = parser.parse_args()

    # Config basic logs
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Define the path
    dir_checkpoint = 'checkpoint/'
    dir_results = 'results/'

    # Report import exceptions raised by parser error
    exception_eval(args)

    # Logging device information
    device = args.device
    logging.info(f'Using device {device}')

    model = DeepCNN(mid_channels=24)
    model.to(torch.device(device))

    train_net(net=model, train_dir=args.train, val_dir=args.val,
              epochs=args.epoch, batches=args.batch, learning_rate=args.lr,
              device=device, save_cp=args.no_cp, save_loss=args.no_curve,
              do_init=args.no_init, do_norm=args.cal_norm)
