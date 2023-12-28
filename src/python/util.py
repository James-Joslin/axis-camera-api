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
    
