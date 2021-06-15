import torch.nn as nn
import torch.nn.functional as F
import torch
from code.modules import *
from scipy.special import gamma, gammaln

####################################################################
######################       Resnet          #######################
####################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, L=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0, in_planes=64, stable_resnet=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        if stable_resnet:
            # Total number of blocks for Stable ResNet
            # https://arxiv.org/pdf/2002.08797.pdf
            L = 0
            for x in num_blocks:
                L+=x
            self.L = L
        else:
            self.L = 1

        self.masks = None

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_planes*8*block.expansion, num_classes)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.L))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out) / self.temp

        return out


#################################################
############# DCTpS ResNets #####################
#################################################


class DCTplusSparseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, L=1, alpha_trainable=True):
        super(DCTplusSparseBasicBlock, self).__init__()
        self.conv1 = DCTplusConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DCTplusConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn2 = nn.BatchNorm2d(planes)

        self.factor = L**(-0.5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                DCTplusConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, alpha_trainable = alpha_trainable),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class DCTplusSparseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, L=1, alpha_trainable=True):
        super(DCTplusSparseBottleneck, self).__init__()
        self.conv1 = DCTplusConv2d(in_planes, planes, kernel_size=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DCTplusConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DCTplusConv2d(planes, self.expansion*planes, kernel_size=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # Normalising factor derived in Stable Resnet paper
        # https://arxiv.org/pdf/2002.08797.pdf
        self.factor = L**(-0.5)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                DCTplusConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, alpha_trainable = alpha_trainable),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out*self.factor + self.shortcut(x)
        out = F.relu(out)
        return out


class DCTplusSparseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, temp=1.0, in_planes=64, stable_resnet=False, alpha_trainable=True):
        super(DCTplusSparseResNet, self).__init__()
        self.in_planes = in_planes
        if stable_resnet:
            # Total number of blocks for Stable ResNet
            # https://arxiv.org/pdf/2002.08797.pdf
            L = 0
            for x in num_blocks:
                L+=x
            self.L = L
        else:
            self.L = 1

        self.masks = None

        self.conv1 = DCTplusConv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, alpha_trainable = alpha_trainable)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2, alpha_trainable = alpha_trainable)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2, alpha_trainable = alpha_trainable)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2, alpha_trainable = alpha_trainable)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DCTplusLinear(in_planes*8*block.expansion, num_classes, alpha_trainable=alpha_trainable)
        self.temp = temp

    def _make_layer(self, block, planes, num_blocks, stride, alpha_trainable = True):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.L, alpha_trainable=alpha_trainable))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out) / self.temp

        return out


# def dctplussparse_resnet50(temp=1.0, **kwargs):
    # model = DCTplusSparseResNet(DCTplusSparseBottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    # return model
# 
# def dctplussparse_resnet18(temp=1.0, **kwargs):
    # model = DCTplusSparseResNet(DCTplusSparseBasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    # return model


def resnet18(temp=1.0, offset='None', alpha_trainable = True, **kwargs):
    if offset =='None':
        model = ResNet(BasicBlock, [2, 2, 2, 2], temp=temp, **kwargs)
    elif offset == 'dct':
        model = DCTplusSparseResNet(DCTplusSparseBasicBlock, [2, 2, 2, 2], temp=temp, alpha_trainable = alpha_trainable, **kwargs)
    else:
        raise NotImplementedError
    return model

def resnet34(temp=1.0, offset='None', alpha_trainable = True, **kwargs):
    if offset =='None':
        model = ResNet(BasicBlock, [3, 4, 6, 3], temp=temp, **kwargs)
    elif offset == 'dct':
        model = DCTplusSparseResNet(DCTplusSparseBasicBlock, [3, 4, 6, 3], temp=temp, alpha_trainable = alpha_trainable, **kwargs)
    else:
        raise NotImplementedError
    return model

def resnet50(temp=1.0, offset='None', alpha_trainable = True, **kwargs):
    if offset == 'None':
        model = ResNet(Bottleneck, [3, 4, 6, 3], temp=temp, **kwargs)
    elif offset == 'dct':
        model = DCTplusSparseResNet(DCTplusSparseBottleneck, [3, 4, 6, 3], temp=temp, alpha_trainable = alpha_trainable, **kwargs)
    else:
        raise NotImplementedError
    return model

def resnet101(temp=1.0, offset='None', alpha_trainable = True, **kwargs):
    if offset == 'None':
        model = ResNet(Bottleneck, [3, 4, 23, 3], temp=temp, **kwargs)
    elif offset == 'dct':
        model = DCTplusSparseResNet(DCTplusSparseBottleneck, [3, 4, 23, 3], temp=temp, alpha_trainable = alpha_trainable, **kwargs)
    else:
        raise NotImplementedError
    return model

def resnet110(temp=1.0, offset='None', alpha_trainable = True, **kwargs):
    if offset=='None':
        model = ResNet(Bottleneck, [3, 4, 26, 3], temp=temp, **kwargs)
    elif offset == 'dct':
        model = DCTplusSparseResNet(DCTplusSparseBottleneck, [3, 4, 26, 3], temp=temp, alpha_trainable = alpha_trainable, **kwargs)
    else:
        raise NotImplementedError
    return model


def resnet152(temp=1.0, offset='None', alpha_trainable = True, **kwargs):
    if offset=='None':
        model = ResNet(Bottleneck, [3, 8, 36, 3], temp=temp, **kwargs)
    elif offset == 'dct':
        model = DCTplusSparseResNet(DCTplusSparseBottleneck, [3, 8, 36, 3], temp=temp, alpha_trainable = alpha_trainable, **kwargs)
    else:
        raise NotImplementedError
    return model



####################################################################
#######################   VGG    ###################################
####################################################################

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(self, dataset='CIFAR10', depth=19, cfg=None, affine=True, batchnorm=True):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm)
        self.dataset = dataset
        if dataset == 'CIFAR10':
            num_classes = 10
        elif dataset == 'CIFAR100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = nn.Linear(cfg[-1], num_classes)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


####################################################################
#######################   DCTpS VGG    #############################
####################################################################



class DCTpluSparseVGG(nn.Module):
    def __init__(self, dataset='CIFAR10', depth=19, cfg=None, affine=True, 
                                    batchnorm=True, alpha_trainable = True):
        super(DCTpluSparseVGG, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]
        self._AFFINE = affine
        self.feature = self.make_layers(cfg, batchnorm, alpha_trainable = alpha_trainable)
        self.dataset = dataset
        if dataset == 'CIFAR10':
            num_classes = 10
        elif dataset == 'CIFAR100':
            num_classes = 100
        elif dataset == 'tiny_imagenet':
            num_classes = 200
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.classifier = DCTplusLinear(cfg[-1], num_classes, bias = True, alpha_trainable = alpha_trainable)

    def make_layers(self, cfg, batch_norm=False, alpha_trainable = True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = DCTplusConv2d(in_channels, v, kernel_size=3, padding=1, bias=False, alpha_trainable = alpha_trainable)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=self._AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.dataset == 'tiny_imagenet':
            x = nn.AvgPool2d(4)(x)
        else:
            x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


def vgg19(dataset = 'CIFAR10', offset='None', alpha_trainable = True):
    if offset=='None':
        return VGG(dataset=dataset, depth=19)
    elif offset == 'dct':
        return DCTpluSparseVGG(dataset=dataset, depth=19,  alpha_trainable =  alpha_trainable)
    else:
        raise NotImplementedError

####################################################################
#######################   Lenet   ###################################
####################################################################

class LeNet(nn.Module):
    def __init__(self, dataset='CIFAR10', bias=False):
        super(LeNet, self).__init__()
        self.dataset = dataset
        if dataset == 'CIFAR10':
            self.num_classes = 10
            self.input_channels = 3
        elif dataset == 'CIFAR100':
            self.num_classes = 100
            self.input_channels = 3
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
            self.input_channels = 3
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.convs = self.make_convs(bias)
        self.fcs = nn.Sequential(
            nn.Linear(in_features=120, out_features=82, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=82, out_features=self.num_classes, bias=bias)
        )


    def make_convs(self, bias):
        layers = []
        layers += [nn.Conv2d(self.input_channels, 6, kernel_size=5, padding=0, bias=bias)]
        layers += [nn.ReLU()]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=bias)]
        layers += [nn.ReLU()]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(16, 120, kernel_size=5, padding=0, bias=bias)]
        layers += [nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        y = self.fcs(x)
        return y


####################################################################
#######################   DCTpS Lenet    #############################
####################################################################


class DCTplusSparseLeNet(nn.Module):
    def __init__(self, dataset='CIFAR10', bias=False, alpha_trainable = True):
        super(DCTplusSparseLeNet, self).__init__()
        self.dataset = dataset
        if dataset == 'CIFAR10':
            self.num_classes = 10
            self.input_channels = 3
        elif dataset == 'CIFAR100':
            self.num_classes = 100
            self.input_channels = 3
        elif dataset == 'tiny_imagenet':
            self.num_classes = 200
            self.input_channels = 3
        else:
            raise NotImplementedError("Unsupported dataset " + dataset)
        self.convs = self.make_convs(bias, alpha_trainable = alpha_trainable)
        self.fcs = nn.Sequential(
            DCTplusLinear(in_features=120, out_features=82, bias=bias, alpha_trainable = alpha_trainable),
            nn.ReLU(),
            DCTplusLinear(in_features=82, out_features=self.num_classes,bias=bias, alpha_trainable = alpha_trainable)
        )


    def make_convs(self, bias, alpha_trainable = True):
        layers = []
        layers += [DCTplusConv2d(self.input_channels, 6, kernel_size=5, padding=0, bias=bias, alpha_trainable = alpha_trainable)]
        layers += [nn.ReLU()]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        layers += [DCTplusConv2d(6, 16, kernel_size=5, padding=0, bias=bias, alpha_trainable = alpha_trainable)]
        layers += [nn.ReLU()]
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        layers += [DCTplusConv2d(16, 120, kernel_size=5, padding=0, bias=bias, alpha_trainable = alpha_trainable)]
        layers += [nn.ReLU()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        y = self.fcs(x)
        return y

def lenet(dataset='CIFAR10', offset='None', alpha_trainable=True):
    if offset == 'None':
        return LeNet(dataset=dataset)
    elif offset == 'dct':
        return DCTplusSparseLeNet(dataset=dataset, alpha_trainable = alpha_trainable)
    else:
        raise NotImplementedError
              
####################################################################
#######################   MobilenetV2   ############################
####################################################################

class MobileBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobileBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),   # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(MobileBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DCTplusSparseMobileBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, alpha_trainable = True):
        super(DCTplusSparseMobileBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = DCTplusConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, alpha_trainable = alpha_trainable)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = DCTplusConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False, alpha_trainable = alpha_trainable)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = DCTplusConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, alpha_trainable = alpha_trainable)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                DCTplusConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, alpha_trainable = alpha_trainable),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class DCTplusSparseMobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),   # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, alpha_trainable = True):
        super(DCTplusSparseMobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = DCTplusConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, alpha_trainable = alpha_trainable)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, alpha_trainable = alpha_trainable)
        self.conv2 = DCTplusConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False, alpha_trainable = alpha_trainable)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = DCTplusLinear(1280, num_classes, alpha_trainable = alpha_trainable)

    def _make_layers(self, in_planes, alpha_trainable = True):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(DCTplusSparseMobileBlock(in_planes, out_planes, expansion, stride, alpha_trainable = alpha_trainable))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def mobilenet(num_classes = 10, offset='None', alpha_trainable=True):
    if offset == 'None':
        return MobileNetV2(num_classes=num_classes)
    elif offset == 'dct':
        return DCTplusSparseMobileNetV2(num_classes=num_classes, alpha_trainable = alpha_trainable)
    else:
        raise NotImplementedError
        
####################################################################
#######################   FixupResNet  #############################
####################################################################

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x



####################################################################
#######################   DCTpS FixupResNet  #######################
####################################################################

def DCTplusSParseConv3x3(in_planes, out_planes, stride=1,alpha_trainable = True):
    """3x3 convolution with padding"""
    return DCTplusConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, alpha_trainable = alpha_trainable)


class DCTplusSparseFixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, alpha_trainable = True):
        super(DCTplusSparseFixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.inplanes = inplanes
        self.planes = planes
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = DCTplusSParseConv3x3(inplanes, planes, stride, alpha_trainable = alpha_trainable)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = DCTplusSParseConv3x3(planes, planes, alpha_trainable = alpha_trainable)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class DCTplusSparseFixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, alpha_trainable = True):
        super(DCTplusSparseFixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = DCTplusSParseConv3x3(3, 16, alpha_trainable = alpha_trainable)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], alpha_trainable = alpha_trainable)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, alpha_trainable = alpha_trainable)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, alpha_trainable = alpha_trainable)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = DCTplusLinear(64, num_classes, alpha_trainable = alpha_trainable)

        for m in self.modules():
            if isinstance(m, DCTplusSparseFixupBasicBlock):
                patch_dim = m.inplanes*3*3
                f = sp.fftpack.dct(np.eye(max(patch_dim, m.planes)), norm='ortho')[:patch_dim, :m.planes]
                stddev = np.sqrt(2 / (m.conv1.conv.weight.shape[0] * np.prod(m.conv1.conv.weight.shape[2:]))) * self.num_layers ** (-0.5)
                target_column_norm = np.sqrt(2)*stddev*(np.exp(gammaln((f.shape[0]+1)/2) - gammaln(f.shape[0]/2) ))
                f_col_norm = np.mean(np.linalg.norm(f, axis = 0))
                m.conv1.scaling.data = torch.tensor(np.sqrt(target_column_norm/f_col_norm))  # mimic the scaling of FixUp resnet weight initialisation
                m.conv2.scaling.data = torch.tensor(0.0)

            elif isinstance(m, DCTplusLinear):
                m.scaling.data = torch.tensor(0.0)  # mimic the scaling of FixUp resnet weight initialisation
                nn.init.constant_(m.linear.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, alpha_trainable = True):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, alpha_trainable = alpha_trainable))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x

def fixup_resnet20(offset='None', alpha_trainable=True, **kwargs):
    """Constructs a Fixup-ResNet-20 model.
    """
    if offset == 'None':
        model = FixupResNet(FixupBasicBlock, [3, 3, 3], **kwargs)
    else:
        model = DCTplusSparseFixupResNet(DCTplusSparseFixupBasicBlock, [3, 3, 3], alpha_trainable = alpha_trainable, **kwargs)
    return model


def fixup_resnet32(offset='None', alpha_trainable=True, **kwargs):
    """Constructs a Fixup-ResNet-32 model.
    """
    if offset=='None':
        model = FixupResNet(FixupBasicBlock, [5, 5, 5], **kwargs)
    else:
        model = DCTplusSparseFixupResNet(DCTplusSparseFixupBasicBlock, [5, 5, 5], alpha_trainable = alpha_trainable, **kwargs)
    return model


def fixup_resnet44(offset='None', alpha_trainable=True, **kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    if offset=='None':
        model = FixupResNet(FixupBasicBlock, [7, 7, 7], **kwargs)
    else:
        model = DCTplusSparseFixupResNet(DCTplusSparseFixupBasicBlock, [7, 7, 7], alpha_trainable = alpha_trainable, **kwargs)
    return model
        
        
def fixup_resnet56(offset='None', alpha_trainable=True, **kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    if offset=='None':
        model = FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
    else:
        model = DCTplusSparseFixupResNet(DCTplusSparseFixupBasicBlock, [9, 9, 9], alpha_trainable = alpha_trainable, **kwargs)
    return model


def fixup_resnet110(offset='None', alpha_trainable=True, **kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    if offset=='None':
        model = FixupResNet(FixupBasicBlock, [18, 18, 18], **kwargs)
    else:
        model = DCTplusSparseFixupResNet(DCTplusSparseFixupBasicBlock, [18, 18, 18], alpha_trainable = alpha_trainable, **kwargs)
        
    return model


def fixup_resnet1202(offset='None', alpha_trainable=True, **kwargs):
    """Constructs a Fixup-ResNet-1202 model.

    """
    if offset=='None':
        model = FixupResNet(FixupBasicBlock, [200, 200, 200], **kwargs)
    else:
        model = DCTplusSparseFixupResNet(DCTplusSparseFixupBasicBlock, [200, 200, 200], alpha_trainable = alpha_trainable, **kwargs)
    return model

