import sys

sys.path.append("..")

import pathlib

import torch.backends.cudnn
import torch.utils.data
from tqdm import tqdm
import torch.nn.functional
from datasets.dataset import RewardingForTest
import matplotlib.pyplot as plt

import argparse, os
import torch.nn as nn
import numpy as np

from model.models import Rewarding

def hyper_args():
    # Test settings
    parser = argparse.ArgumentParser(description="PyTorch Corss-modality Registration Rewarding Model")
    datasetName = 'M3FD'
    # parser.add_argument('--ir', default=fr'D:\Yang.Z.Learn\dataset\final_dataset\{datasetName}\train\deformIR', type=pathlib.Path) # train: sigma=32 test: sigma=32
    parser.add_argument('--rgb', default=fr'./{datasetName}/test/RGB', type=pathlib.Path)
    parser.add_argument('--REG', default=fr'./{datasetName}/test/REG', type=pathlib.Path)
    parser.add_argument('--savepath', default=fr'../result/{datasetName}')
    parser.add_argument('--savepath_npy', default=fr'../{datasetName}')
    parser.add_argument("--batchsize", type=int, default=1, help="training batch size")
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--pretrained", default=f"./cache/cp_0350.pth", type=str, help="path to pretrained model (default: none)")
    # Please replace 'checkpoint_path' with a yourself filename
    parser.add_argument("--ckpt", default=f"./checkpoint/", type=str, help="path to pretrained model (default: none)")
    args = parser.parse_args()
    return args

def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")

    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    data = RewardingForTest(args.rgb, args.reg)
    testing_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True) # , pin_memory=True

    print("===> Building model")
    net = Rewarding()
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # TODO: optionally copy weights from a checkpoint
    if args.pretrained != "none":
        if os.path.isfile(args.pretrained):
            print("=> loading model '{}'".format(args.pretrained))
            model_state_dict = torch.load(args.pretrained)
            net.load_state_dict(model_state_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained))
            return

    print("===> Testing")
    net.eval()
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)
    if not os.path.exists(args.savepath_npy):
        os.makedirs(args.savepath_npy)
    tqdm_loader = tqdm(testing_data_loader, disable=True)
    for (reg,rgb, name) in tqdm_loader:
        print("process a batch....")
        reg = reg.cuda()
        rgb = rgb.cuda()
        input_ = torch.cat((reg, rgb), dim=1)
        x = net(input_)
        x = x.cpu().detach().numpy()
        for i in range(x.shape[0]):
            plt.clf()
            plt.cla()
            heapmap = x[i][0]
            # 绘制热图
            plt.imshow(heapmap, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title('Heatmap')
            plt.savefig(f'{args.savepath}/{name[i]}.png')



if __name__ == '__main__':
    args = hyper_args()
    main(args)
    print("Done")