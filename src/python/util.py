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
import matplotlib.patches as patches
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

import datetime
import glob
import re

import torch
from torch.utils.data import Dataset

import json
import os
import cv2

from tqdm import tqdm

class Preprocessor:
    def __init__(self, image_directories, annotations_path, processed_image_dir, processed_annotations_dir, target_size=(600, 480)):
        self.image_directories = image_directories
        self.annotations_path = annotations_path
        self.processed_image_dir = processed_image_dir 
        self.processed_annotations_dir = processed_annotations_dir
        self.target_width, self.target_height = target_size

        if not os.path.exists(self.processed_image_dir):
            os.makedirs(self.processed_image_dir)

    def read_annotations(self, file_path):
        annotations = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line.strip())
                annotations.append(data)
        return annotations

    def preprocess_images(self):
        annotations = self.read_annotations(self.annotations_path)
        all_fboxes = []

        for annotation in tqdm(annotations):
            image_id = annotation['ID']
            image_file = self.find_image(image_id)
            if image_file:
                image = cv2.imread(image_file)
                processed_image, adjusted_annotations = self.transform(image, annotation)
                # self.plot_image_with_boxes(processed_image, adjusted_annotations, image_id, './src/python/temp/')
                all_fboxes.append({
                    'ID': image_id, 
                    'persons': [box['fbox'] for box in adjusted_annotations['persons']],
                    'masks': [box['fbox'] for box in adjusted_annotations['masks']]
                })
                # Optionally save the processed image
                # self.save_image(image_id, processed_image)

        # Save all fboxes to a new JSON file
        self.save_fboxes_json(all_fboxes, "annotations")

    def find_image(self, image_id):
        for directory in self.image_directories:
            for filename in os.listdir(directory):
                if filename.startswith(image_id):
                    return os.path.join(directory, filename)
        return None

    def transform(self, image, annotation):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_height, orig_width, _ = image.shape
        # boxes = annotation['gtboxes']

        # Resize the image while maintaining aspect ratio
        scale = min(self.target_width / orig_width, self.target_height / orig_height)
        new_width, new_height = int(orig_width * scale), int(orig_height * scale)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Calculate padding to maintain aspect ratio
        pad_x = (self.target_width - new_width) // 2
        pad_y = (self.target_height - new_height) // 2

        # Pad the resized image to the target dimensions
        processed_image = cv2.copyMakeBorder(resized_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        person_boxes = [self.adjust_bbox(box['fbox'], scale, pad_x, pad_y) for box in annotation['gtboxes'] if box['tag'] == 'person']
        mask_boxes = [self.adjust_bbox(box['fbox'], scale, pad_x, pad_y) for box in annotation['gtboxes'] if box['tag'] == 'mask']

        # Update the annotations with adjusted bounding boxes
        adjusted_annotations = {
            'persons': [{'fbox': box} for box in person_boxes],
            'masks': [{'fbox': box} for box in mask_boxes]
        }

        return processed_image, adjusted_annotations

    def adjust_bbox(self, bbox, scale, pad_x, pad_y):
        x, y, w, h = bbox
        x = int(x * scale + pad_x)
        y = int(y * scale + pad_y)
        w = int(w * scale)
        h = int(h * scale)
        return [x, y, w, h]

    def plot_image_with_boxes(self, image, adjusted_annotations, image_id, save_dir):
        # Create the save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Loop through each bounding box and add a rectangle to the plot
        for bbox in adjusted_annotations['persons']:
            fbox = bbox['fbox']
            rect = patches.Rectangle((fbox[0], fbox[1]), fbox[2], fbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # Plot 'mask' bounding boxes
        for bbox in adjusted_annotations['masks']:
            fbox = bbox['fbox']
            rect = patches.Rectangle((fbox[0], fbox[1]), fbox[2], fbox[3], linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

        # Define the path for the plot
        plot_path = os.path.join(save_dir, f"{image_id}.png")

        # Save the figure
        plt.title(f"Image ID: {image_id}")
        plt.axis('off')  # Optional: turn off the axis for a cleaner image
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

    def save_fboxes_json(self, all_fboxes, file_name):
        path = os.path.join(self.processed_annotations_dir, file_name)
        with open(f'{path}.json', 'w') as file:
            json.dump(all_fboxes, file, indent=4)
        file.close()

    def save_image(self, image_id, processed_image):
        # Convert RGB back to BGR for saving
        processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        # Use the image ID from the annotations as the filename
        new_filename = f"{image_id}.jpg"  # You can add a prefix or suffix if desired
        new_path = os.path.join(self.processed_image_dir, new_filename)

        # Save the processed image
        cv2.imwrite(new_path, processed_image_bgr)

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

def preprocess_image(image_path, target_width=600, target_height=480):
    # Read the image
    image = cv2.imread(image_path)
    
    # Calculate the aspect ratio
    h, w, _ = image.shape
    scale = min(target_width/w, target_height/h)
    
    # Resize the image
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a new image with the target size and put the resized image into it
    preprocessed = cv2.copyMakeBorder(resized, 
                                      top=(target_height - new_h) // 2, 
                                      bottom=(target_height - new_h) // 2, 
                                      left=(target_width - new_w) // 2, 
                                      right=(target_width - new_w) // 2, 
                                      borderType=cv2.BORDER_CONSTANT, 
                                      value=[0, 0, 0])  # Black padding
    
    return preprocessed

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
