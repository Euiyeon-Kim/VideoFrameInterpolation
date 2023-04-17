import os
import time
import shutil
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import data as benchmarks
from Trainer import Trainer
from utils.logger import Logger
from utils.scheduler import get_lr, set_lr
from utils.env import get_options, prepare_env
from evaluate import validate_vimeo90k, validate_ucf101


def train(args, trainer):
    local_rank = args.local_rank

    # Preparation for training
    if local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        shutil.copy(args.config, os.path.join(args.log_dir, 'config.yaml'))
        summary_writer = SummaryWriter(args.log_dir)
        logger = Logger(summary_writer, metric_summary_freq=args.metric_summary_freq)

        # Print options
        print(args)

        # Print model information
        print(trainer.model.module)
        num_params = sum(p.numel() for p in trainer.model.parameters())
        print('Number of params:', num_params)

    # Prepare training
    step = 0
    start_epoch = 0
    best_psnr = 0.
    if args.resume:
        assert os.path.exists(args.resume), f"{args.resume} should exists"
        chkpt = torch.load(args.resume)
        if 'step' in chkpt.keys():
            step = chkpt['step']
            start_epoch = chkpt['epoch']
            best_psnr = chkpt['best_psnr']
        trainer.load_trained(chkpt)
    last_lr_decay_iter = args.last_lr_decay_iter

    # Build Dataloader
    train_dataset = getattr(benchmarks, f'{args.data_name}')(args)
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, sampler=sampler)

    trainer.train()
    for cur_epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(cur_epoch)

        time_stamp = time.time()
        for i, batch in enumerate(train_dataloader):
            # Data preparation
            for k, v in batch.items():
                batch[k] = v.to(args.device, non_blocking=True)

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            cur_lr = get_lr(args, step, last_lr_decay_iter)

            # Update model weight
            set_img_dict = local_rank == 0 and (step + 1) % args.img_summary_freq == 0
            metrics = trainer.one_step(batch, cur_lr, set_img_dict)

            train_time_interval = time.time() - time_stamp
            step += 1

            # Logging
            if local_rank == 0:
                metrics.update({
                    'lr': cur_lr,
                    'data_time': data_time_interval,
                    'train_time': train_time_interval,
                })
                logger.push(metrics)

                # Image logging
                if step % args.img_summary_freq == 0:
                    img_dict = trainer.get_img_dict()
                    logger.add_image_summary(img_dict)

                # Save model weighs frequently with optimizer
                if step % args.save_latest_freq == 0:
                    latest_path = f'{args.log_dir}/latest.pth'
                    trainer.save_model(latest_path, cur_epoch, step, best_psnr, save_optim=True)

        if local_rank == 0:
            if (cur_epoch + 1) % args.save_every_freq_epoch == 0:
                checkpoint_path = f'{args.log_dir}/epoch_{cur_epoch+1:03d}.pth'
                trainer.save_model(checkpoint_path, cur_epoch + 1, step, best_psnr, save_optim=True)

            if (cur_epoch + 1) % args.valid_freq_epoch == 0:
                trainer.eval()
                val_results = {}

                if not args.val_datasets:
                    continue

                if 'vimeo90k' in args.val_datasets:
                    results_dict = trainer.validate_vimeo90k()
                    val_results.update(results_dict)

                if 'ucf101' in args.val_datasets:
                    results_dict = trainer.validate_ucf101()
                    val_results.update(results_dict)

                cur_psnr = val_results[f"val/{args.save_best_benchmark}_psnr"]

                # Save best model
                if cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    best_path = f'{args.log_dir}/best_{args.save_best_benchmark}.pth'
                    trainer.save_model(best_path, cur_epoch + 1, step, best_psnr, save_optim=False)

                logger.write_dict(val_results, step=cur_epoch + 1)
                print(f"Epoch {cur_epoch + 1} Validation Done - Best: {best_psnr:.3f}")

                trainer.train()

        dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EuiyeonKim VFIs')
    parser.add_argument('--exp_name', default='debug', type=str)
    parser.add_argument('--config', type=str, default='configs/IFRNet.yaml', help='Configuration YAML path')
    parser.add_argument('--resume', type=str)
    parsed = parser.parse_args()

    # Set Environment - distributed training
    args = get_options(parser.parse_args())
    prepare_env(args)
    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer(args, args.local_rank)

    train(args, trainer)

    dist.destroy_process_group()
