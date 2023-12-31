import json
import os
import glob
import util
import cv2
import matplotlib.pyplot as plt
import numpy as np
from network_config import mobileNetConfig
from torch.utils.data import DataLoader
import torch
from torch import optim

if __name__ == "__main__":
    mobileNet_config = mobileNetConfig()
    util.set_seed(mobileNet_config.SEED)
    
    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    
    
