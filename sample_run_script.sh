#!/bin/bash

python  ../main.py  --dataset CIFAR100 \
						--network_name resnet50 \
						--init zeros \
						--offset dct \
						--opt adam \
						--init_lr 0.001 \
						--rigl \
						--prune_method 5 \
						--pruning_factor 0.01 \
						--iterations 3 \
						--alpha_trainable True \
						--save_results True \
						--save_model True
