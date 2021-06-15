import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy as sp
import scipy.fftpack

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

#############################################
############## DCTpS Layers #################
#############################################

class DCTconv2d(nn.Conv2d):

  def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = False):
    super(DCTconv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=bias)
    patch_dim = int(in_channels/groups)*kernel_size*kernel_size
    f = sp.fftpack.dct(np.eye(max(patch_dim, out_channels)), norm='ortho')[:patch_dim, :out_channels]
    kernel_shape = (kernel_size,kernel_size,int(in_channels/groups),out_channels)
    self.weight.data = torch.from_numpy(np.transpose(np.reshape(f,kernel_shape), [3,2,0,1])).float()

    self.weight.requires_grad = False


class DCT(nn.Linear):

  def __init__(self, in_features, out_features, bias = False):
    super(DCT, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
    w0 = sp.fftpack.dct(np.eye(max(in_features, out_features)), norm = 'ortho')[:in_features,:out_features]

    self.weight.data = torch.from_numpy(w0).float().transpose(0,1)

    self.weight.requires_grad = False




class DCTplusConv2d(nn.Module):

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding  = 0, groups = 1, bias = False, alpha_trainable = True):
        super(DCTplusConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, groups = groups, bias=bias)
        self.dct = DCTconv2d(in_planes, planes, kernel_size, stride, padding, groups = groups, bias = False)
        scaling = torch.tensor(1.0)
        self.scaling = nn.Parameter(scaling, requires_grad=alpha_trainable)

    def forward(self, x):
        return self.scaling*self.dct(x) + self.conv(x)


class DCTplusLinear(nn.Module):

    def __init__(self, in_features, out_features, bias = True, alpha_trainable = True):
        super(DCTplusLinear, self).__init__()
        print("alpha_trainable:", alpha_trainable)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.dct = DCT(in_features, out_features, bias = False)
        scaling = torch.tensor(1.0)
        self.scaling = nn.Parameter(scaling, requires_grad=alpha_trainable)

    def forward(self, x):
        return self.scaling*self.dct(x) + self.linear(x)



