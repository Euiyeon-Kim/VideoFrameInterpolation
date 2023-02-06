import random

import numpy as np

import torch

from data.base import BaseDataset, read


class Vimeo90K(BaseDataset):
    def __init__(self, args, is_train=True):
        super(Vimeo90K, self).__init__(args, is_train)
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
            f01 = read(f"{self.flow_paths[idx]}/{'flow_t0.flo' if self.distill_bwd else 'flow_01.npy'}")
            f10 = read(f"{self.flow_paths[idx]}/{'flow_t1.flo' if self.distill_bwd else 'flow_10.npy'}")
        else:
            x0_path = f'{self.paths[idx]}/im3.png'
            x1_path = f'{self.paths[idx]}/im1.png'
            f01 = read(f"{self.flow_paths[idx]}/{'flow_t1.flo' if self.distill_bwd else 'flow_10.npy'}")
            f10 = read(f"{self.flow_paths[idx]}/{'flow_t0.flo' if self.distill_bwd else 'flow_01.npy'}")

        xt_path = f'{self.paths[idx]}/im2.png'
        x0 = read(x0_path)
        xt = read(xt_path)
        x1 = read(x1_path)
        frames = np.concatenate([x0, x1, xt], axis=2)
        frames = np.concatenate([frames, f01, f10], axis=2)

        frames = self.transform(frames)
        t = np.expand_dims(np.array(0.5, dtype=np.float32), 0)

        x0, x1, xt, f01, f10 = torch.split(frames, [3, 3, 3, 2, 2], dim=0)
        return {
            'x0': x0,
            'x1': x1,
            'xt': xt,
            't': t,
            'f01': f01,
            'f10': f10,
        }

    def get_test_item(self, idx):
        x0 = read(f'{self.paths[idx]}/im1.png')
        xt = read(f'{self.paths[idx]}/im2.png')
        x1 = read(f'{self.paths[idx]}/im3.png')
        frames = np.stack([x0, x1, xt], axis=0)

        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))

        x0, x1, xt = torch.chunk(frames, 3, dim=0)
        t = torch.from_numpy(np.expand_dims(np.array(0.5, dtype=np.float32), 0))

        return {
            'x0': x0[0],
            'x1': x1[0],
            'xt': xt[0],
            't': t,
        }

    def __getitem__(self, idx):
        if self.is_train:
            return self.get_train_item(idx)
        else:
            return self.get_test_item(idx)



