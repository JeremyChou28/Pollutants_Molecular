"""
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2024-09-27 13:31:36
"""

import os
import random
import numpy as np
import torch


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


def str2bool(str):
    return True if str.lower() == "true" else False
