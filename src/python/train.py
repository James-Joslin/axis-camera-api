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
    annotations_dir = secrets['imageAnnotations']
    # util.string_to_json(annotations_dir)
    
    images_array = []
    ground_truth = []
    images_dir = secrets['imagePixelData']
    for i, anno_set in enumerate(glob.glob("./src/python/*.json")):
        with open(anno_set, 'r') as file:
            annos = json.load(file)
            for ele in annos:
                image = cv2.imread(os.path.join(images_dir, ele['image']))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                # plt.imshow(image)
                image = np.transpose(image, (2, 0, 1))
                bbox = np.array(ele['boxes']).astype(np.float32)
                
            # Transpose bounding box coordinates
                bbox = bbox[:, [1, 0, 3, 2]]  # Swap x and y coordinates
                
                valid_boxes = []
                for box in bbox:
                    if box[2] > box[0] and box[3] > box[1]:
                        valid_boxes.append(box)
                    else:
                        print(f"Invalid box removed: {box} in file {anno_set}")

                images_array.append(image)
                ground_truth.append(np.array(valid_boxes))
                
        file.close()
    
    images_array = np.array(images_array).astype(np.float32) / 255.
    indices = np.random.permutation(images_array.shape[0])
    shuffled_images = images_array# [indices]
    shuffled_ground_truth = ground_truth #[ground_truth[i] for i in indices]
    print(shuffled_ground_truth[0])

    train_images, train_gt, val_images, val_gt, test_images, test_gt = util.split_data(0.7, 0.2, 0.1, shuffled_images, shuffled_ground_truth)
    print(train_gt[0])
    train_data = util.CustomDataset(train_images, train_gt)
    val_data = util.CustomDataset(val_images, val_gt)
    test_data = util.CustomDataset(test_images, test_gt)
    train_loader = DataLoader(train_data, batch_size=mobileNet_config.BATCH, shuffle=False, collate_fn=util.collate_fn)
    val_loader = DataLoader(val_data, batch_size=mobileNet_config.BATCH, shuffle=False, collate_fn=util.collate_fn)
    test_loader = DataLoader(test_data, batch_size=mobileNet_config.BATCH, shuffle=False, collate_fn=util.collate_fn)

    # Create the model
    model = util.create_mobilenetv2_ssd(mobileNet_config.NUM_CLASSES, image_size = mobileNet_config.IMG_SIZE)
    model.to(torch.device(mobileNet_config.DEVICE))
    dummy_input = torch.randn(1, 3, 480, 640).to(torch.device(mobileNet_config.DEVICE))
    with torch.no_grad():
        features = model.backbone(dummy_input)

    # Print the shapes of all feature maps in the FPN output
        for level, feature in features.items():
            print(f"Shape of FPN output at level {level}: {feature.shape}")

        # If you want to forward pass through the first layer of the SSD head for a specific feature level
        # choose one level from the FPN output, for example '0', '1', or '2'
        selected_level = '0'  # Choose based on your model's specifics
        if selected_level in features:
            first_ssd_layer_output = model.head.classification_head.module_list[0](features[selected_level])
        print(f"Shape of first SSD layer output: {first_ssd_layer_output.shape}")

    # Define the loss functions
    classification_loss = torch.nn.CrossEntropyLoss()
    localization_loss = torch.nn.SmoothL1Loss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=mobileNet_config.LR)

    for epoch in range(mobileNet_config.EPOCHS):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(torch.device(mobileNet_config.DEVICE))
            print(images.shape)
            targets = [{k: v.to(torch.device(mobileNet_config.DEVICE)) for k, v in t.items()} for t in targets]
            print(targets)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(images, targets) # object detection models require data and targets during training

            # Calculate the loss
            loss_c = classification_loss(output['scores'], targets['labels'])
            loss_l = localization_loss(output['boxes'], targets['boxes'])
            loss = loss_c + loss_l

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{mobileNet_config.EPOCHS}], Loss: {epoch_loss:.4f}")

        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                targets = [{k: v.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) for k, v in t.items()} for t in targets]

                # Forward pass
                output = model(images)

                # Calculate the loss
                loss_c = classification_loss(output['scores'], targets['labels'])
                loss_l = localization_loss(output['boxes'], targets['boxes'])
                loss = loss_c + loss_l

                val_running_loss += loss.item()

        val_epoch_loss = val_running_loss / len(val_loader)
        print(f"Validation Loss: {val_epoch_loss:.4f}")

