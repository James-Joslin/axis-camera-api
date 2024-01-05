import json
import util

from torch.utils.data import DataLoader

from network_config import SqueezeNetConfig

from squeezenet_architecture import ObjectDetector

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
    val_loader = DataLoader(val_builder, batch_size=1, shuffle=True, num_workers=2, collate_fn=util.collate_fn)

    # model
    model = ObjectDetector().to(squeezeNet_config.DEVICE)
    util.model_summary(model, (3, 360, 640))
    
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

        rpn_logits, rpn_bbox_regs = model(images)
        print(rpn_logits)
        print(rpn_bbox_regs)
        print(device_annotations)
        pass