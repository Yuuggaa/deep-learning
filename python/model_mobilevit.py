import timm
import torch.nn as nn

def create_mobilevit(num_classes):
    model = timm.create_model(
        "mobilevit_xxs",
        pretrained=True,
        num_classes=num_classes
    )
    return model
