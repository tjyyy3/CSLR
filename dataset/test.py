import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler


a = np.load(f"/home/czk/SLProject/VAC-chinese/preprocess/phoenix2014/test_info.npy", allow_pickle=True).item()
c = np.load(f"/home/czk/SLProject/VAC-chinese/preprocess/SLR_dataset/train_info.npy", allow_pickle=True).item()
b = 1
print("a")