import json
import util

from torch.utils.data import DataLoader

from network_config import SqueezeNetConfig

from squeezenet_architecture import ObjectDetector, generate_anchors, apply_deltas_to_anchors

import torch

def test_anchor_generation():
    print("testing anchor outputs")
    # Generate anchors
    anchors = generate_anchors(base_size=16, ratios=[0.5, 0.75, 1, 1.25, 1.5, 2], scales=[64, 128, 256, 512, 1024, 2048])
    print("Generated anchors shape:", anchors.shape)

    # Assert the number of anchors
    assert anchors.shape[0] == 36, "Total number of anchors does not match model's expectation."

def test_apply_deltas_to_anchors():
    print("testing delta outputs")
    # Generate some dummy deltas (similar to what your RPN would output)
    dummy_deltas = torch.randn(1, 36, 14, 14, 4)  # Shape: [batch, anchors, height, width, 4]
    
    # Generate anchors
    anchors = generate_anchors(base_size=16, ratios=[0.5, 0.75, 1, 1.25, 1.5, 2], scales=[64, 128, 256, 512, 1024, 2048])
    
    # Apply deltas to anchors (Assuming you have a function like 'apply_deltas_to_anchors')
    decoded_boxes = apply_deltas_to_anchors(anchors, dummy_deltas)
    print("Decoded boxes shape:", decoded_boxes.shape)

if __name__ == "__main__":
    # Config
    squeezeNet_config = SqueezeNetConfig()
    
    # Seed
    util.set_seed(squeezeNet_config.SEED)
    
    # Secrets - file directories
    with open('./secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    
    # Load training data
    train_builder = util.DatasetBuilder(
        secrets["PostProcessData"]["imagesTrain"],
        secrets["PostProcessData"]["annosTrain"]    
    )
    train_loader = DataLoader(train_builder, batch_size=2, shuffle=True, num_workers=2, collate_fn=util.collate_fn)
    
    # Load val data
    val_builder = util.DatasetBuilder(
        secrets["PostProcessData"]["imagesVal"],
        secrets["PostProcessData"]["annosVal"]   
    )
    val_loader = DataLoader(val_builder, batch_size=2, shuffle=True, num_workers=2, collate_fn=util.collate_fn)

    # model
    model = ObjectDetector().to(squeezeNet_config.DEVICE)
    # util.model_summary(model, (3, 360, 640))
    
    # training loop
    for images, batch_annotations in train_loader:
        images = images.to(squeezeNet_config.DEVICE)  # Move images to the device
        
        # Initialize a new list to hold dictionaries with tensors moved to the device
        device_annotations = []

        # Iterate through each annotations dictionary in the batch
        for annot in batch_annotations:
            # Move each tensor in the dictionary to the device
            device_annot = {k: v.to(squeezeNet_config.DEVICE) for k, v in annot.items()}
            device_annotations.append(device_annot)

        # for item in device_annotations:
        #     print(item)

        rpn_probs, rpn_bbox_regs = model.forward(images)
        anchors = generate_anchors(squeezeNet_config.ANCHOR_BASE_SIZE, squeezeNet_config.ANCHOR_RATIO, squeezeNet_config.ANCHOR_SCALE)
        print("Generated anchors shape:", anchors.shape)
        decoded_boxes = [apply_deltas_to_anchors(anchors, bbox_reg) for bbox_reg in rpn_bbox_regs]

        pass