import random

import numpy as np

import torch

from imageio import imread
from data.base import BaseDataset, BaseDatasetwFlow, read


class Vimeo90K(BaseDataset):
    def __init__(self, args, is_train=True):
        super(Vimeo90K, self).__init__(args, is_train)
        info_file_path = f"{self.root}/tri_trainlist.txt" if is_train else f"{self.root}/tri_testlist.txt"
        with open(info_file_path, 'r') as f:
            for line in f:
                l = line.strip()
                if len(l) != 0:
                    self.paths.append(f"{self.root}/sequences/{l}")

    def get_train_item(self, idx):
        if random.randint(0, 1):
            x0_path = f'{self.paths[idx]}/im1.png'
            x1_path = f'{self.paths[idx]}/im3.png'
        else:
            x0_path = f'{self.paths[idx]}/im3.png'
            x1_path = f'{self.paths[idx]}/im1.png'

        xt_path = f'{self.paths[idx]}/im2.png'
        x0 = imread(x0_path)
        xt = imread(xt_path)
        x1 = imread(x1_path)
        frames = np.concatenate([x0, x1, xt], axis=2)

        frames = self.transform(frames) / 255.
        x0, x1, xt = torch.split(frames, [3, 3, 3], dim=0)
        t = np.expand_dims(np.array(0.5, dtype=np.float32), (0, 1, 2))
        return {
            'x0': x0,
            'x1': x1,
            'xt': xt,
            't': t,
        }

    def get_test_item(self, idx):
        x0 = imread(f'{self.paths[idx]}/im1.png')
        xt = imread(f'{self.paths[idx]}/im2.png')
        x1 = imread(f'{self.paths[idx]}/im3.png')

        frames = np.concatenate([x0, x1, xt], axis=2)
        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames.transpose(2, 0, 1)) / 255.
        x0, x1, xt = torch.chunk(frames, 3, dim=0)
        t = torch.from_numpy(np.expand_dims(np.array(0.5, dtype=np.float32), (0, 1, 2)))
        return {
            'x0': x0,
            'x1': x1,
            'xt': xt,
            't': t,
        }

    def __getitem__(self, idx):
        if self.is_train:
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)


class Vimeo90KwFlow(BaseDatasetwFlow):
    def __init__(self, args, is_train=True):
        super(Vimeo90KwFlow, self).__init__(args, is_train)
        self.distill_bwd = args.distill_bwd
        info_file_path = f"{self.root}/tri_trainlist.txt" if is_train else f"{self.root}/tri_testlist.txt"
        with open(info_file_path, 'r') as f:
            for line in f:
                l = line.strip()
                if len(l) != 0:
                    self.paths.append(f"{self.root}/sequences/{l}")
                    self.flow_paths.append(f"{self.root}/{self.flow_dir}/{l}")

    def get_train_item(self, idx):
        if random.randint(0, 1):
            x0_path = f'{self.paths[idx]}/im1.png'
            x1_path = f'{self.paths[idx]}/im3.png'
            f0x = read(f"{self.flow_paths[idx]}/{'flow_t0.flo' if self.distill_bwd else 'flow_01.npy'}")
            f1x = read(f"{self.flow_paths[idx]}/{'flow_t1.flo' if self.distill_bwd else 'flow_10.npy'}")
        else:
            x0_path = f'{self.paths[idx]}/im3.png'
            x1_path = f'{self.paths[idx]}/im1.png'
            f0x = read(f"{self.flow_paths[idx]}/{'flow_t1.flo' if self.distill_bwd else 'flow_10.npy'}")
            f1x = read(f"{self.flow_paths[idx]}/{'flow_t0.flo' if self.distill_bwd else 'flow_01.npy'}")

        xt_path = f'{self.paths[idx]}/im2.png'
        x0 = imread(x0_path)
        xt = imread(xt_path)
        x1 = imread(x1_path)
        frames = np.concatenate([x0, x1, xt], axis=2)
        frames = np.concatenate([frames, f0x, f1x], axis=2)

        frames = self.transform(frames) / 255.
        t = np.expand_dims(np.array(0.5, dtype=np.float32), (0, 1, 2))

        x0, x1, xt, f0x, f1x = torch.split(frames, [3, 3, 3, 2, 2], dim=0)
        return {
            'x0': x0,
            'x1': x1,
            'xt': xt,
            't': t,
            'f0x': f0x,
            'f1x': f1x,
        }

    def get_test_item(self, idx):
        x0 = imread(f'{self.paths[idx]}/im1.png')
        xt = imread(f'{self.paths[idx]}/im2.png')
        x1 = imread(f'{self.paths[idx]}/im3.png')

        frames = np.concatenate([x0, x1, xt], axis=2)
        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames.transpose(2, 0, 1)) / 255.
        x0, x1, xt = torch.chunk(frames, 3, dim=0)
        t = torch.from_numpy(np.expand_dims(np.array(0.5, dtype=np.float32), (0, 1, 2)))
        return {
            'x0': x0,
            'x1': x1,
            'xt': xt,
            't': t,
        }

    def __getitem__(self, idx):
        if self.is_train:
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)

