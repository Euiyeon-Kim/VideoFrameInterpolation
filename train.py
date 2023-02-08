import os
import time
import shutil
import argparse

import numpy as np
import oyaml as yaml
from dotmap import DotMap

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
import models
import data as benchmarks
from utils.logger import Logger
from evaluate import validate_vimeo90k


def get_lr(args, iters):
    ratio = 0.5 * (1.0 + np.cos(iters / (args.num_epochs * args.iters_per_epoch) * np.pi))
    lr = (args.start_lr - args.end_lr) * ratio + args.end_lr
    return lr


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args, ddp_model):
    local_rank = args.local_rank
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(args.log_dir, 'config.yaml'))
        summary_writer = SummaryWriter(args.log_dir)
        logger = Logger(summary_writer, metric_summary_freq=args.metric_summary_freq)
        print(ddp_model.module)
        num_params = sum(p.numel() for p in ddp_model.parameters())
        print('Number of params:', num_params)

    dataset_train = getattr(benchmarks, f'{args.data_name}')(args)
    sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, sampler=sampler)

    args.iters_per_epoch = len(dataloader_train)
    iters = args.resume_epoch * args.iters_per_epoch

    optimizer = optim.AdamW(ddp_model.parameters(), lr=args.start_lr, weight_decay=0)

    best_psnr = 0.0
    for epoch in range(args.resume_epoch, args.num_epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(dataloader_train):
            # Data preparation
            for k, v in batch.items():
                batch[k] = v.to(args.device)
            time_stamp = time.time()

            # Set lr
            cur_lr = get_lr(args, iters)
            set_lr(optimizer, cur_lr)

            # Init gradient
            optimizer.zero_grad()

            # Forwarding
            log_dict, total_loss, metrics = ddp_model(batch)

            # Optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)
            optimizer.step()

            # Logging
            train_time_interval = time.time() - time_stamp
            if local_rank == 0:
                metrics.update({
                    'lr': cur_lr,
                    'time': train_time_interval,
                })
                logger.push(metrics)

                # Image logging
                if (iters + 1) % args.img_summary_freq == 0:
                    img_dict = ddp_model.module.get_log_dict(batch, log_dict)
                    logger.add_image_summary(img_dict)

                # Save model weighs frequently with optimizer
                if (iters + 1) % args.save_latest_freq == 0:
                    checkpoint_path = f'{args.log_dir}/latest.pth'
                    torch.save({
                        'model': ddp_model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': iters + 1,
                        'best_psnr': best_psnr,
                    }, checkpoint_path)

            iters += 1

        if (epoch + 1) % args.valid_freq_epoch == 0 and local_rank == 0:
            val_results = {}

            if 'vimeo90k' in args.val_datasets:
                results_dict = validate_vimeo90k(args, model)
                val_results.update(results_dict)

            cur_psnr = val_results[f"{args.save_best_benchmark}_psnr"]

            # Save best model
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                checkpoint_path = f'{args.log_dir}/best_{args.save_best_benchmark}.pth'
                torch.save({
                    'model': ddp_model.module.state_dict()
                }, checkpoint_path)

            logger.write_dict(val_results, step=epoch+1)
            print(f"Epoch {epoch + 1} Validation Done - Best: {best_psnr:.3f}")
            model.train()

        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuiyeonKim VFIs')
    parser.add_argument('--exp_name', default='debug', type=str)
    parser.add_argument('--config', type=str, default='configs/IFRNet.yaml', help='Configuration YAML path')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)

    # Parse Args from Configs
    parsed = parser.parse_args()
    with open(parsed.config, 'r') as f:
        config = yaml.safe_load(f)
    args = DotMap(config)
    args.config = parsed.config
    args.exp_name = parsed.exp_name
    args.world_size = parsed.world_size
    args.local_rank = parsed.local_rank
    args.log_dir = os.path.join('exps', args.exp_name)
    args.num_workers = args.batch_size

    # Set Environment - distributed training
    dist.init_process_group(backend='nccl', world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    # Set Environment - seed & optimization
    utils.set_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = True

    # Build Model
    model = getattr(models, f'{args.model_name}')(args).to(args.device)

    # Load state dict
    if args.load_gmflow:
        print(f"Load GMFlow weight from {args.load_gmflow}")
        checkpoint = torch.load(args.load_gmflow, map_location='cpu')['model']
        model.load_state_dict(checkpoint, strict=False)

    ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    train(args, ddp_model)

    dist.destroy_process_group()
