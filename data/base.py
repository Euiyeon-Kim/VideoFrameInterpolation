from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.crop_size = [args.crop_h, args.crop_w]
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

        # Flip augmentation - horizontal
        if random.random() < 0.5:
            frames = frames[:, ::-1, :]

        # Flip augmentation - vertical
        if random.random() < 0.5:
            frames = frames[::-1, :, :]

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
