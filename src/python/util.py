import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import numpy as np
from itertools import product
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import random
import ast

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

import datetime
import glob
import re

import torch
from torch.utils.data import Dataset

import torchvision.models.detection
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator

# Models
def create_mobilenetv2_ssd(num_classes, image_size=(480, 640)):
    # Load pre-trained MobileNetV2
    dummy_input = torch.randn(1, 3, 480, 640)  # Batch size of 1, 3 color channels, 480x640 image

    weights = MobileNet_V2_Weights.DEFAULT
    mobilenetv2_backbone = mobilenet_v2(weights=weights).features
    backbone_out_channels = mobilenetv2_backbone[-1].out_channels
    
    with torch.no_grad():  # We don't need to compute gradients for this
        for i, layer in enumerate(mobilenetv2_backbone):
            dummy_input = layer(dummy_input)
            if i in [4, 10]:  # Layers of interest
                print(f"Output shape after layer {i}: {dummy_input.shape}")
                
    # Freeze the early layers of the model
    for param in mobilenetv2_backbone[:14].parameters():
        param.requires_grad = False

    backbone = BackboneWithFPN(
        backbone=mobilenetv2_backbone,
        return_layers={'4': '0', '10': '1', '18': '2'},
        in_channels_list=[32, 64, 1280],  # Updated channel sizes
        out_channels=backbone_out_channels
    )

    # Define the anchor generator
    anchor_sizes = ((32,), (64,), (128,), (256,))  # Add an additional size for the 'pool' level
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)  # Ensure this matches the number of feature maps

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Create the SSD head with the number of classes
    ssd_head = torchvision.models.detection.ssd.SSDHead(
        in_channels=[backbone.out_channels],
        num_anchors=anchor_generator.num_anchors_per_location(),
        num_classes=num_classes
    )

    # Create the full SSD model
    # Define image normalization means and stds
    image_mean = [0.485, 0.456, 0.406]  # Commonly used ImageNet mean
    image_std = [0.229, 0.224, 0.225]  # Commonly used ImageNet std

    # Create the full SSD model
    model = torchvision.models.detection.ssd.SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=image_size,
        num_classes=num_classes,
        image_mean=image_mean,
        image_std=image_std,
        head=ssd_head,  # If you created an SSDHead separately, otherwise set to None and the default head will be used
        score_thresh=0.01,
        nms_thresh=0.45,
        detections_per_img=200,
        iou_thresh=0.5,
        topk_candidates=400,
        positive_fraction=0.25
        # Add any other specific kwargs as needed
    )
    # print(model)
    return model

# Data input
def string_to_json(parent_dir: str):
    for i, annos in enumerate(glob.glob(f'{parent_dir}*')):
        all_data = []  # Initialize an empty list to hold all data
        
        # Read the annotation file
        with open(annos, 'r') as file:
            for line in file.readlines():
                # Split the line into the image path and the bounding boxes
                path, boxes_str = line.strip().split(': ')
                
                ints = [int(s) for s in re.findall(r'\b\d+\b', boxes_str)]
                res = [ints[i:i+4] for i in range(0, len(ints), 4)]  # Convert to list of lists for JSON compatibility
                
                # Create a dictionary for this image
                image_data = {
                    'image': path.strip('"'),
                    'boxes': res
                }
                
                # Add the dictionary to the list of all data
                all_data.append(image_data)
        file.close()

        # Convert the list of data to JSON format
        json_data = json.dumps(all_data, indent=4)

        # Write the JSON data to a new file
        with open(f'./src/python/annotations_data{i+1}.json', 'w') as json_file:
            json_file.write(json_data)
        json_file.close()

# Dataloading
def split_data(train_size:float, val_size:float, test_size:float, x_data:np.array, y_data):
    
    assert 1 - (train_size + val_size + test_size) < 1e-6
    assert len(x_data) == len(y_data)
    
    train_end_idx = int(train_size * len(x_data))
    val_end_idx = int((train_size + val_size) * len(y_data))

    train_images = x_data[:train_end_idx]
    train_gt = y_data[:train_end_idx]

    val_images = x_data[train_end_idx:val_end_idx]
    val_gt = y_data[train_end_idx:val_end_idx]

    test_images = x_data[val_end_idx:]
    test_gt = y_data[val_end_idx:]
    
    return train_images, train_gt, val_images, val_gt, test_images, test_gt

class CustomDataset(Dataset):
    def __init__(self, images, ground_truths):
        self.images = images  # A list or array of images
        self.ground_truths = ground_truths  # A corresponding list of bounding box arrays

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert images and bounding boxes to PyTorch tensors
        image = torch.tensor(self.images[idx], dtype=torch.float)
        bboxes = torch.tensor(self.ground_truths[idx], dtype=torch.float)
        
        # Construct the target dictionary
        target = {}
        target['boxes'] = bboxes
        target['labels'] = torch.ones((len(bboxes),), dtype=torch.int64)  # Assuming all objects are class 1
        
        return image, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Convert lists of tensors to a single tensor for the images
    images = torch.stack(images, dim=0)

    return images, targets

# Output and export
def export_to_onnx(onnx_base_path, onnx_name, model:nn.Module, checkpoint, input_size):
    # write brackets model
    model.load_state_dict(
        checkpoint['model_state_dict']
    )
    print(model)
    x = torch.randn(1, 1, input_size, requires_grad=False)
    torch_out = model(x)
    torch.onnx.export(model,                   # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f'{onnx_base_path}{onnx_name}.onnx',   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    print(f'File saved to: {onnx_base_path}{onnx_name}')

# System checks
def check_gpu():
    if torch.cuda.is_available():
        f"GPU is available. Detected {torch.cuda.device_count()} GPU(s)."
        return "cuda"
    else:
        "GPU is not available."
        return "cpu"
    
def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  # For consistent results on the GPU
    torch.backends.cudnn.benchmark = False  # Faster convolutions, but might introduce randomness
    
# Evaluate Functions
def model_summary(model, input_size):
    print("----------------------------------------------------------------")
    print(f"Input Shape:               {str(input_size).ljust(25)}")
    print("----------------------------------------------------------------")
    print("Layer (type)               Output Shape         Param #")
    print("================================================================")
    
    total_params = 0

    def register_hook(module):
        def hook(module, input, output):
            nonlocal total_params
            num_params = sum(p.numel() for p in module.parameters())
            total_params += num_params

            # Remove torch.Size
            if isinstance(output, tuple):
                output_shape = [str(list(o.shape)) if torch.is_tensor(o) else str(type(o)) for o in output]
                # Pick first size if there are multiple identical sizes in the tuple
                output_shape = output_shape[0]
            else:
                output_shape = str(list(output.shape))

            if len(list(module.named_children())) == 0:  # Only print leaf nodes
                print(f"{module.__class__.__name__.ljust(25)}  {output_shape.ljust(25)} {f'{num_params:,}'}")

        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    model.apply(register_hook)

    print("----------------------------------------------------------------")
    DEVICE = next(model.parameters()).device
    output = model(torch.randn(1, *input_size).to(DEVICE))

    for h in hooks:
        h.remove()

    output_shape = str(list(output.shape)) if torch.is_tensor(output) else str(type(output))
    print("----------------------------------------------------------------")
    print(f"Total params: {total_params:,}")
    print(f"Output Shape: {output_shape.ljust(25)}")
    print(f'Model on: {next(model.parameters()).device}')
    print("----------------------------------------------------------------")

def save_checkpoint(current_epoch, model, optimiser, loss, checkpoint_file='./'):
    torch.save({
        'ganme' : current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss,
    }, checkpoint_file)
    print(f'Saving model at epoch {current_epoch}')
    
def load_checkpoint(directory):
    if os.path.isfile(directory):
        print("Loading Model Checkpoint")
        checkpoint = torch.load(directory)
        return checkpoint
    else:
        pass
    
