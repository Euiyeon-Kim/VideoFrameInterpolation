import math

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data


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
def validate_vimeo90k(args, model):
    psnr_list = []
    # ssim_list = []
    eval_results = {}

    model.eval()
    val_dataset = data.Vimeo90K(args, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=16,
                                pin_memory=True, shuffle=False, drop_last=False)
    print('Number of validation images: %d' % len(val_dataset))

    for batch in val_dataloader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        gt_01 = batch['xt'] / 255.
        pred = model(batch)

        b = pred.shape[0]
        for i in range(b):
            psnr = calculate_psnr(pred[i], gt_01[i]).detach().cpu().numpy()
            psnr_list.append(psnr)
            # ssim = calculate_ssim(pred, gt_01).detach().cpu().numpy()
            # ssim_list.append(ssim)

    final_psnr = np.mean(psnr_list)
    print(f"Validation Vimeo90K PSNR: {final_psnr:.4f}")
    # final_ssim = np.mean(ssim_list)
    # print(f"Validation Vimeo90K PSNR: {final_psnr:.4f}, SSIM: {final_ssim:.4f}")

    eval_results['vimeo90k_psnr'] = final_psnr
    # eval_results['vimeo90k_ssim'] = final_ssim

    return eval_results


if __name__ == '__main__':
    import argparse
    import oyaml as yaml
    from dotmap import DotMap

    parser = argparse.ArgumentParser(description='EuiyeonKim VFIs')
    parser.add_argument('--exp_name', default='DAT/debug', type=str)
    parser.add_argument('--test_epoch', type=int)
    parsed = parser.parse_args()
    with open(parsed.config, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
