import re
import random

import cv2
import numpy as np
from imageio import imread

import torch
from torch.utils.data import Dataset


def readFloat(name):
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape) == 3:
            return data[:,:,0:3]
        else:
            return data
    return imread(name)


def read(file):
    if file.endswith('.float3'): return readFloat(file)
    elif file.endswith('.flo'): return readFlow(file)
    elif file.endswith('.ppm'): return readImage(file)
    elif file.endswith('.pgm'): return readImage(file)
    elif file.endswith('.png'): return readImage(file)
    elif file.endswith('.jpg'): return readImage(file)
    elif file.endswith('.npy'): return np.load(file)
    elif file.endswith('.pfm'): return readPFM(file)[0]
    else: raise Exception('don\'t know how to read %s' % file)


class BaseDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train
        self.crop_size = [args.crop_h, args.crop_w]

        self.root = args.root
        self.paths = []

        # self.flow_dir = args.flow_dir
        # self.flow_paths = []

    def transform(self, frames):
        # Random resizing
        if random.uniform(0, 1) < 0.1:
            img0 = cv2.resize(frames[:, :, :3], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(frames[:, :, 3:6], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            imgt = cv2.resize(frames[:, :, 6:9], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            frames = np.concatenate([img0, img1, imgt], axis=2)
            # flow = cv2.resize(frames[:, :, 9:], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
            # frames = np.concatenate([img0, img1, imgt, flow], axis=2)

        h, w, _ = frames.shape

        # Random cropping augmentation
        h_offset = random.choice(range(h - self.crop_size[0] + 1))
        w_offset = random.choice(range(w - self.crop_size[1] + 1))
        frames = frames[h_offset:h_offset + self.crop_size[0], w_offset: w_offset + self.crop_size[1], :]

        # Random reverse channel
        if random.random() < 0.5:
            img0 = frames[:, :, :3]
            img1 = frames[:, :, 3:6]
            imgt = frames[:, :, 6:9]
            frames[:, :, :3] = img0[:, :, ::-1]
            frames[:, :, 3:6] = img1[:, :, ::-1]
            frames[:, :, 6:9] = imgt[:, :, ::-1]

        # Flip augmentation - vertical
        if random.random() < 0.5:
            frames = frames[::-1, :, :]
            # flow = np.concatenate((frames[:, :, 9:10], -frames[:, :, 10:11],
            #                        frames[:, :, 11:12], -frames[:, :, 12:13]), 2)
            # frames[:, :, 9:] = flow

        # Flip augmentation - horizontal
        if random.random() < 0.5:
            frames = frames[:, ::-1, :]
            # flow = np.concatenate((-frames[:, :, 9:10], frames[:, :, 10:11],
            #                        -frames[:, :, 11:12], frames[:, :, 12:13]), 2)
            # frames[:, :, 9:] = flow

        rot = random.randint(0, 3)
        frames = np.rot90(frames, rot, (0, 1))
        # if rot == 1:
        #     flows = np.concatenate((frames[:, :, 10:11], -frames[:, :, 9:10],
        #                             frames[:, :, 12:13], -frames[:, :, 11:12]), axis=-1)
        #     frames[:, :, 9:] = flows
        # if rot == 2:
        #     frames[:, :, 9:] = -frames[:, :, 9:]
        # if rot == 3:
        #     flows = np.concatenate((-frames[:, :, 10:11], frames[:, :, 9:10],
        #                             -frames[:, :, 12:13], frames[:, :, 11:12]), axis=-1)
        #     frames[:, :, 9:] = flows

        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames.transpose(2, 0, 1))

        return frames

    def __getitem__(self, idx):
        return self.paths[idx]

    def __len__(self):
        return len(self.paths)


class BaseDatasetwFlow(Dataset):
    def __init__(self, args, is_train=True):
        self.args = args
        self.is_train = is_train
        self.crop_size = [args.crop_h, args.crop_w]

        self.root = args.root
        self.paths = []

        self.flow_dir = args.flow_dir
        self.flow_paths = []

    def transform(self, frames):
        # Random resizing
        if random.uniform(0, 1) < 0.1:
            img0 = cv2.resize(frames[:, :, :3], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            img1 = cv2.resize(frames[:, :, 3:6], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            imgt = cv2.resize(frames[:, :, 6:9], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(frames[:, :, 9:], dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
            frames = np.concatenate([img0, img1, imgt, flow], axis=2)

        h, w, _ = frames.shape

        # Random cropping augmentation
        h_offset = random.choice(range(h - self.crop_size[0] + 1))
        w_offset = random.choice(range(w - self.crop_size[1] + 1))
        frames = frames[h_offset:h_offset + self.crop_size[0], w_offset: w_offset + self.crop_size[1], :]

        # Random reverse channel
        if random.random() < 0.5:
            img0 = frames[:, :, :3]
            img1 = frames[:, :, 3:6]
            imgt = frames[:, :, 6:9]
            frames[:, :, :3] = img0[:, :, ::-1]
            frames[:, :, 3:6] = img1[:, :, ::-1]
            frames[:, :, 6:9] = imgt[:, :, ::-1]

        # Flip augmentation - vertical
        if random.random() < 0.5:
            frames = frames[::-1, :, :]
            flow = np.concatenate((frames[:, :, 9:10], -frames[:, :, 10:11],
                                   frames[:, :, 11:12], -frames[:, :, 12:13]), 2)
            frames[:, :, 9:] = flow

        # Flip augmentation - horizontal
        if random.random() < 0.5:
            frames = frames[:, ::-1, :]
            flow = np.concatenate((-frames[:, :, 9:10], frames[:, :, 10:11],
                                   -frames[:, :, 11:12], frames[:, :, 12:13]), 2)
            frames[:, :, 9:] = flow

        rot = random.randint(0, 3)
        frames = np.rot90(frames, rot, (0, 1))
        if rot == 1:
            flows = np.concatenate((frames[:, :, 10:11], -frames[:, :, 9:10],
                                    frames[:, :, 12:13], -frames[:, :, 11:12]), axis=-1)
            frames[:, :, 9:] = flows
        if rot == 2:
            frames[:, :, 9:] = -frames[:, :, 9:]
        if rot == 3:
            flows = np.concatenate((-frames[:, :, 10:11], frames[:, :, 9:10],
                                    -frames[:, :, 12:13], frames[:, :, 11:12]), axis=-1)
            frames[:, :, 9:] = flows

        frames = frames.astype(np.float32)
        frames = torch.from_numpy(frames.transpose(2, 0, 1))

        return frames

    def __getitem__(self, idx):
        return self.paths[idx]

    def __len__(self):
        return len(self.paths)
