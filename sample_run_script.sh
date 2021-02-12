#!/bin/bash

python  ./main.py   --dataset CIFAR100 \
                    --iterations 1 \
                    --network_name dctplussparse_resnet50 --init zeros \
                    --opt adam \
                    --prune_method 5 \
                    --pruning_factor 0.001 \
                    --init_lr 0.001 \
