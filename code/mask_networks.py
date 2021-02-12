# masking function from FORCE implementation: https://github.com/naver/force

import torch.nn as nn
import numpy as np

def apply_prune_mask(net, keep_masks):
    """
    Function that takes a network and a list of masks and applies it to the relevant layers.
    mask[i] == 0 --> Prune parameter
    mask[i] == 1 --> Keep parameter
    """

    prunable_layers = filter(
        lambda layer: (type(layer) == nn.Conv2d) or (type(layer) == nn.Linear), net.modules())


    hook_handlers = []
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """
            def hook(grads):
                return grads * keep_mask

            return hook

        # Step 1: Set the masked weights to zero (Biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        h = layer.weight.register_hook(hook_factory(keep_mask))
        hook_handlers.append(h)

    return hook_handlers
