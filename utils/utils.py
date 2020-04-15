import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_manual_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
