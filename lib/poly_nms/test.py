import numpy as np
import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from poly_nms import poly_nms  # noqa: E402