import torch.nn as nn

from models.resnet_pretrain import Pretrain_Wide_ResNet_Model, Pretrain_ResNet18_Model, ResNet18
from models.custom_resnet import resnet18
import models.transform_layers as TL

def get_simclr_augmentation(P, image_size):

    # parameter for resizecrop
    resize_scale = (P.resize_factor, 1.0) # resize scaling factor
    if P.resize_fix: # if resize_fix is True, use same scale
        resize_scale = (P.resize_factor, P.resize_factor)

    # Align augmentation
    color_jitter = TL.ColorJitterLayer(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    color_gray = TL.RandomColorGrayLayer(p=0.2)
    resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=image_size)

    # Transform define #
    if P.dataset == 'imagenet': # Using RandomResizedCrop at PIL transform
        transform = nn.Sequential(
            color_jitter,
            color_gray,
        )
    else:
        transform = nn.Sequential(
            color_jitter,
            color_gray,
            resize_crop,
        )

    return transform


def get_shift_classifer(model, shift_head_neuron):
    model.shift_cls_layer = nn.Linear(model.last_dim, shift_head_neuron)
    return model


def get_classifier(mode, n_classes=10, activation='relu', freezing_layer=133):
    if mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes, activation=activation)
    elif mode == "pretrain-wide-resnet":
        classifier = Pretrain_Wide_ResNet_Model(num_classes=n_classes)
    elif mode == 'resnet18':
        classifier = ResNet18(num_classes=n_classes)
    elif mode == 'pretrain-resnet18':
        classifier = Pretrain_ResNet18_Model(num_classes=n_classes)
    elif mode == 'custome_resnet18':
        classifier = resnet18(num_classes=n_classes)
    else:
        raise NotImplementedError()

    return classifier

