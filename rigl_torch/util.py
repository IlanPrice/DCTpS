import torch
import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy as sp
import scipy.fftpack

import code


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, code.modules.DCT, code.modules.DCTconv2d, )


def get_weighted_layers(model, i=0, layers=None, linear_layers_mask=None):
    # print("model:", model)
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        #print(p)
        if type(p) == torch.nn.Linear:
            layers.append([p])
            linear_layers_mask.append(1)

        elif type(p) not in EXCLUDED_TYPES:
            #print(type(p))
            #print("------------")
            if hasattr(p, 'weight'):
                layers.append([p])
                linear_layers_mask.append(0)
            elif (type(p) == torchvision.models.resnet.Bottleneck) or (type(p) == torchvision.models.resnet.BasicBlock):
                _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)
            else:
                _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)

    return layers, linear_layers_mask, i 



def get_W(model, return_linear_layers_mask=False):
    layers, linear_layers_mask, _ = get_weighted_layers(model)
    
    # print("layers:", layers)

    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W
