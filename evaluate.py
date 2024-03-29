import os
import math

import numpy as np
from tqdm import tqdm
from imageio import imread

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
import models


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().cuda()
    return window


def calculate_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def calculate_psnr(img1, img2):
    psnr = -10 * torch.log10(((img1 - img2) * (img1 - img2)).mean())
    return psnr


@torch.no_grad()
def validate_vimeo90k(args, model, batch_size=1, report_ssim=False):
    psnr_list, ssim_list = [], []
    eval_results = {}

    model.eval()
    val_dataset = data.Vimeo90K(args, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16,
                                pin_memory=True, shuffle=False, drop_last=False)
    print('Number of validation images: %d' % len(val_dataset))

    for batch in val_dataloader:
        x0 = batch['x0'].cuda()
        x1 = batch['x1'].cuda()
        xt = batch['xt'].cuda()
        t = batch['t'].cuda()
        pred = model.inference(x0, x1, t)

        b = pred.shape[0]
        for i in range(b):
            psnr = calculate_psnr(pred[i], xt[i]).detach().cpu().numpy()
            psnr_list.append(psnr)
            if report_ssim:
                ssim = calculate_ssim(pred, xt).detach().cpu().numpy()
                ssim_list.append(ssim)

    final_psnr = np.mean(psnr_list)
    log_txt = f"Validation Vimeo90K PSNR: {final_psnr:.4f}"
    eval_results['val/vimeo90k_psnr'] = final_psnr

    if report_ssim:
        final_ssim = np.mean(ssim_list)
        eval_results['val/vimeo90k_ssim'] = final_ssim
        log_txt = log_txt + f", SSIM: {final_ssim:.4f}"

    print(log_txt)
    return eval_results


@torch.no_grad()
def validate_ucf101(model, report_ssim=False):
    psnr_list, ssim_list = [], []
    eval_results = {}
    ucf_path = 'datasets/UCF-101/test'
    dirs = os.listdir(ucf_path)
    print('Number of validation images: %d' % len(dirs))

    t = torch.ones((1, 1, 1, 1)).cuda().float() * 0.5
    for d in tqdm(dirs):
        img0 = (ucf_path + '/' + d + '/frame_00.png')
        img1 = (ucf_path + '/' + d + '/frame_02.png')
        gt = (ucf_path + '/' + d + '/frame_01_gt.png')
        img0 = (torch.tensor(imread(img0).transpose(2, 0, 1))).cuda().float().unsqueeze(0) / 255.
        img1 = (torch.tensor(imread(img1).transpose(2, 0, 1))).cuda().float().unsqueeze(0) / 255.
        gt_01 = (torch.tensor(imread(gt).transpose(2, 0, 1))).cuda().float().unsqueeze(0) / 255.
        pred = model.inference(img0, img1, t)
        psnr = calculate_psnr(pred, gt_01).detach().cpu().numpy()
        psnr_list.append(psnr)
        if report_ssim:
            ssim = calculate_ssim(pred, gt_01).detach().cpu().numpy()
            ssim_list.append(ssim)

    final_psnr = np.mean(psnr_list)
    eval_results['val/ucf101_psnr'] = final_psnr
    log_txt = f"Validation UCF101 PSNR: {final_psnr:.4f}"

    if report_ssim:
        final_ssim = np.mean(ssim_list)
        log_txt = log_txt + f", SSIM: {final_ssim:.4f}"
        eval_results['val/ucf101_ssim'] = final_ssim

    print(log_txt)
    return eval_results


@torch.no_grad()
def validate_snu(model, report_ssim=False):
    eval_results = {}
    snu_path = 'datasets/SNU-FILM'
    t = torch.ones((1, 1, 1, 1)).cuda().float() * 0.5
    level_list = ['test-easy.txt', 'test-medium.txt', 'test-hard.txt', 'test-extreme.txt']

    for test_file in level_list:
        psnr_list, ssim_list = [], []
        file_list = []

        with open(os.path.join(snu_path, test_file), "r") as f:
            for line in f:
                line = line.strip()
                file_list.append(line.split(' '))

        for line in tqdm(file_list, desc=test_file[:-4]):
            I0_path = line[0].replace('data', 'datasets')
            I1_path = line[1].replace('data', 'datasets')
            I2_path = line[2].replace('data', 'datasets')
            I0 = (torch.tensor(imread(I0_path).transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
            gt_01 = (torch.tensor(imread(I1_path).transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()
            I2 = (torch.tensor(imread(I2_path).transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).cuda()

            padder = data.InputPadder(I0.shape, divisor=16)
            I0, I2 = padder.pad(I0, I2)

            pred = model.inference(I0, I2, t)
            pred = padder.unpad(pred)

            psnr = calculate_psnr(pred, gt_01).detach().cpu().numpy()
            psnr_list.append(psnr)
            if report_ssim:
                ssim = calculate_ssim(pred, gt_01).detach().cpu().numpy()
                ssim_list.append(ssim)

        final_psnr = np.mean(psnr_list)
        log_txt = f"Validation SNU-FILM {test_file[:-4]} PSNR: {final_psnr:.4f}"
        eval_results[f'val/snu_{test_file[:-4]}_psnr'] = final_psnr

        if report_ssim:
            final_ssim = np.mean(ssim_list)
            log_txt = log_txt + f"SSIM: {final_ssim:.4f}"
            eval_results[f'val/snu_{test_file[:-4]}_psnr'] = final_psnr

        print(log_txt)

    return eval_results


if __name__ == '__main__':
    import argparse
    import oyaml as yaml
    from dotmap import DotMap

    parser = argparse.ArgumentParser(description='EuiyeonKim VFIs evaluation')
    parser.add_argument('--exp_name',
                        default='DCNDAT/DCNDATv1_shareDCNBwarpEmbT_QDCNAttnBothDAT_noPE_E5D10_distill_dim64_p256_bwarp', type=str)
    parser.add_argument('--test_epoch', type=int)
    parsed = parser.parse_args()
    config_path = f'exps/{parsed.exp_name}/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
    args.exp_name = parsed.exp_name
    if parsed.test_epoch:
        ckpt_path = f'exps/{parsed.exp_name}/epoch_{parsed.test_epoch:03d}.pth'
    else:
        ckpt_path = f'exps/{parsed.exp_name}/best_{args.save_best_benchmark}.pth'

    # Model definition
    model = getattr(models, f'{args.model_name}')(args).cuda().eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load weights
    ckpt = torch.load(ckpt_path)['model']
    model.load_state_dict(ckpt, strict=True)
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    eval_dict = validate_vimeo90k(args, model, batch_size=4)
    # eval_dict = validate_ucf101(args, model)
    # eval_dict = validate_snu(args, model)
