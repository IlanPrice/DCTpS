# DCT plus Sparse (DCTpS)

This code base is for running the experiments in "Dense for the Price of Sparse: Improving performance of sparsely initialised networks via a subspace offset" (Price & Tanner, ICML2021), available at http://arxiv.org/abs/2102.07655.

One can select DCTpS or standard versions of the architectures reported in the paper:

- ResNet50
- VGG19
- Lenet-5
- MobileNetV2
- FixupResNet110
- Resnet18

and others as well (e.g. other Resnets, FixupResnets, etc.)

In the code, the Prune-at-Initialization methods/sparse support allocation methods are referred to by number:

1. FORCE
3. Random (Uniform)
4. Random (EPF)
5. Random (EPL)
6. Random (ERK)

To run RigL, the `--rigl` flag must be included in the run script, though this in only currently supported for uniform, EPL, and ERK support distributions.

Note: Experiments testing SynFlow were run separately with the authors' published code (1)

## Setup

Create a conda environment with the necessary dependencies with `conda env create -f environment.yml`.

## Run

Experiments may be run by calling `main.py` with the appropriate arguments, e.g. to train a DCTpS ResNet50 on CIFAR100 with adam at 0.1% density:

```
python  ./main.py   --dataset CIFAR100 \
                    --iterations 1 \
                    --network_name resnet50 \
                    --offset dct \
                    --init zeros \
                    --opt adam \
                    --prune_method 5 \
                    --pruning_factor 0.001 \
                    --init_lr 0.001 \
```

Two things to remember when running experiments with DCTpS networks is to add specify the offset to be `dct` ahead of the network name and to include `--init zeros`, because the sparse trainable matrices in DCTpS networks are zero-initialized.


## Acknowledgements

The experimental setup in this code was originally adapted from code published by the authors of the recent paper (2), and takes the implementations of MobileNetv2 and Fixup ResNets from their respective public implementations (3, 4). The RigL implementation was adapted from the its pytorch implementation (5).

_______


(1)  https://github.com/ganguli-lab/Synaptic-Flow

(2) "Progressive Skeletonization: Trimming more fat from a network at initialization" by De Jorge et al. See https://github.com/naver/force. 

(3) https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py

(4) https://github.com/hongyi-zhang/Fixup

(5) https://pypi.org/project/rigl-torch/
