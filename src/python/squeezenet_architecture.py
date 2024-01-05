from network_config import SqueezeNetConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            self.inner_blocks.append(nn.Conv2d(in_channels, out_channels, 1))
            self.layer_blocks.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, x):
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))

        for feature, inner_block, layer_block in zip(
            reversed(x[:-1]), reversed(self.inner_blocks[:-1]), reversed(self.layer_blocks[:-1])
        ):
            # Upsample 'last_inner' to match the spatial dimensions of 'inner_lateral'
            inner_lateral = inner_block(feature)
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[2:], mode="nearest")
            
            # Element-wise addition
            last_inner = inner_lateral + inner_top_down
            
            # Pass the combined features through the next layer block and add to results
            results.insert(0, layer_block(last_inner))

        return results

class FireWithBypass(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireWithBypass, self).__init__()
        # Adjust squeeze channels based on the squeeze ratio of 0.75
        adjusted_squeeze_channels = int(squeeze_channels * 0.75)

        self.squeeze = nn.Conv2d(in_channels, adjusted_squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        # Expand layers
        self.expand1x1 = nn.Conv2d(adjusted_squeeze_channels, expand_channels, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(adjusted_squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Store the original input for the bypass connection
        identity = x

        x = self.squeeze_activation(self.squeeze(x))
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

        # Add the bypass (identity) connection if the sizes match
        if x.size() == identity.size():
            x += identity
        return x

class CustomSqueezeNet(nn.Module):
    def __init__(self):
        super(CustomSqueezeNet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2)  # Adjusted for smaller models
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        # Fire modules with simple bypass and squeeze ratio adjustments
        self.fire2 = FireWithBypass(64, 16, 64)
        self.fire3 = FireWithBypass(128, 16, 64)
        self.fire4 = FireWithBypass(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire5 = FireWithBypass(256, 32, 128)
        self.fire6 = FireWithBypass(256, 48, 192)
        self.fire7 = FireWithBypass(384, 48, 192)
        self.fire8 = FireWithBypass(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.fire9 = FireWithBypass(512, 64, 256)
        
        # Final convolution (no softmax)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=1)  # The output channels can be adjusted

        self.fpn = FPN(in_channels_list=[256, 256, 384, 512], out_channels=256)
        
    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        f2 = self.fire2(x)
        f3 = self.fire3(f2)
        f4 = self.fire4(f3)
        x = self.maxpool2(f4)
        f5 = self.fire5(x)
        f6 = self.fire6(f5)
        f7 = self.fire7(f6)
        f8 = self.fire8(f7)
        x = self.maxpool3(f8)
        f9 = self.fire9(x)

        # Collect selected feature maps
        feature_maps = [f4, f5, f7, f9]

        # Pass feature maps through FPN
        fpn_feature_maps = self.fpn(feature_maps)

        return fpn_feature_maps

class RPN(nn.Module):
    def __init__(self, anchor_scales, anchor_ratios, feature_channels, mid_channels=256):
        super(RPN, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        
        # The number of anchors
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        # The RPN has a conv layer followed by two separate layers for classifying and regressing anchors
        self.conv = nn.Conv2d(feature_channels, mid_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, self.num_anchors * 2, 1)  # 2 for objectness score (obj, not obj)
        self.bbox_pred = nn.Conv2d(mid_channels, self.num_anchors * 4, 1)  # 4 for bbox offsets

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize the weights and biases with a specific strategy
        pass

    def forward(self, features):
        # Apply the RPN layers to get the objectness score and bbox predictions
        x = self.conv(features)
        logits = self.cls_logits(x)
        bbox_regs = self.bbox_pred(x)
        return logits, bbox_regs

# Example of initializing the RPN
# feature_channels = 512  # Adjust based on the output channels of your SqueezeNet
# rpn = RPN(anchor_scales=[128, 256, 512], anchor_ratios=[0.5, 1, 2], feature_channels=feature_channels)
    
class ObjectDetector(nn.Module):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        self.backbone = CustomSqueezeNet()  # Updated backbone
        self.rpn = RPN(anchor_scales=[128, 256, 512], anchor_ratios=[0.5, 1, 2], feature_channels=256)  # Update channels if needed

    def forward(self, x):
        features = self.backbone(x)
        # Assume RPN takes a list of feature maps
        rpn_logits = []
        rpn_bbox_regs = []
        for feature in features:
            logits, bbox_regs = self.rpn(feature)
            rpn_logits.append(logits)
            rpn_bbox_regs.append(bbox_regs)
        return rpn_logits, rpn_bbox_regs
