import timm
import torch.nn as nn

def create_swin(num_classes):
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes
    )
    return model
