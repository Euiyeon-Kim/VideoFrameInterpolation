import time
import oyaml as yaml
from dotmap import DotMap

import torch
import torch.nn as nn
import torch.nn.functional as F

import models


CONFIG = 'configs/DAT.yaml'
with open(CONFIG, 'r') as f:
    config = yaml.safe_load(f)
args = DotMap(config)


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

img0 = torch.randn(1, 3, 256, 448).cuda()
img1 = torch.randn(1, 3, 256, 448).cuda()
embt = torch.tensor(1/2).float().view(1, 1).cuda()
inp_dict = {
    'x0': img0,
    'x1': img1,
    't': embt,
}

# Build Model
model = getattr(models, f'{args.model_name}')(args).cuda().eval()

with torch.no_grad():
    for i in range(100):
        out = model(inp_dict)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(100):
        out = model(inp_dict)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print('Time: {:.3f}s'.format((time.time() - time_stamp) / 100))

total = sum([param.nelement() for param in model.parameters()])
print('Parameters: {:.2f}M'.format(total / 1e6))