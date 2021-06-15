
import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from code.pruning_algos import prune, iterative_pruning
from code.mask_networks import apply_prune_mask
from code.utils import *
from code.datasets import *

import os
import argparse
import random
import json

import shutil
import copy
import distutils
import distutils.util

from rigl_torch.RigL import RigLScheduler

def parseArgs():

    parser = argparse.ArgumentParser(
                description="Training CIFAR / Tiny-Imagenet.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pruning_factor", type=float, default=0.01, dest="pruning_factor",
                        help='Global fraction of connections after pruning')

    parser.add_argument("--prune_method", type=int, default=5, dest="prune_method",
                        help="""Which pruning method to use:
                                1->FORCE
                                3->Uniform Random
                                4->equal_per_filter
                                5->equal_per_layer
                                6->Erdos reini kernel
                                """)
    parser.add_argument("--rigl", action = 'store_true', default = False, help = 'Flag to specify the user of RigL for dynamic sparses training. Only works with prune methods 3, 5, and 6')

    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        dest="dataset_name", help='Dataset to train on')

    parser.add_argument("--offset", type=str, default = 'None', dest ="offset", help = 'Offset from the origin - currently must be one of "None" or "dct"')

    parser.add_argument("--alpha_trainable", default = True, type=lambda x:bool(distutils.util.strtobool(x)), help = 'Boolean flag for whether or not alpha is trainable in DCTpS layers')
    
    parser.add_argument("--network_name", type=str, default='resnet50', dest="network_name",
                        help='Model to train')

    parser.add_argument("--num_steps", type=int, default=10,
                        help='Number of steps to use with iterative pruning')

    parser.add_argument("--mode", type=str, default='exp',
                        help='Mode of creating the iterative pruning steps one of "linear" or "exp".')

    parser.add_argument("--num_batches", type=int, default=1,
                        help='Number of batches to be used during FORCE pruning')

    parser.add_argument("--save_loc", type=str, default='../saved_models/',
                        dest="save_loc", help='Path where to save the model')

    parser.add_argument("--opt", type=str, default='sgd',
                        dest="optimiser",
                        help='Choice of optimisation algorithm - options are SGD and adam')

    parser.add_argument("--init_lr", type=float, default = 0.1,
                        dest="init_lr",
                        help='Choice of initial learning rate')

    parser.add_argument("--lr_schedule", action = 'store_true', default = False,
                        help="Indicates that lr should decrease by the speacified factor gamma")

    parser.add_argument("--frac-train-data", type=float, default=0.9, dest="frac_data_for_train",
                        help='Fraction of data used for training')

    parser.add_argument("--init", type=str, default='normal_kaiming',
                        help='Which initialization method to use for the trainable weight tensors')

    parser.add_argument("--in_planes", type=int, default=64,
                        help="Number of input planes in Resnet. Afterwards they duplicate after each conv with stride 2 as usual.")

    parser.add_argument("--stable_resnet", action = 'store_true', default = False)
    
    parser.add_argument('--augment_data', default = True, type=lambda x:bool(distutils.util.strtobool(x)),
                        help="Whether or not data is augmented with random flips, etc")

    parser.add_argument('--centre_and_scale', default = True, type=lambda x:bool(distutils.util.strtobool(x)),
                           help="Whether or not data is normalised")

    parser.add_argument("--iterations", type=int, default=1, dest="iterations",
                    help="How many random seed runs to perform")

    parser.add_argument("--lr_decay_rate", type=float, default = 0.1,
                        dest="gamma",
                        help = 'Factor multiplied by learning rate at specified epoch milestones if --lr_schedule is True')

    parser.add_argument("--weight_decay", type=float, default = 0.0005,
                        help = 'weight decay in optimiser')

    parser.add_argument("--batch_size", type=int, default = 128,
                            help = 'train batch size')

    parser.add_argument("--test_batch_size", type=int, default = 256,
                            help = 'test batch size')

    parser.add_argument("--epochs", type=int, default = 200,
                            help = 'Number of training epochs')
                            
    parser.add_argument("--milestones", type=float, nargs = "*", default = [120, 160],
                        help='Epochs at which to shrink learning rate. Only applicable if args.lr_schedule == True')

    parser.add_argument('--save_model', default = True, type=lambda x:bool(distutils.util.strtobool(x)),
                            help="Whether or not to save the trained model, etc")
    parser.add_argument('--save_results', default = True, type=lambda x:bool(distutils.util.strtobool(x)),
                           help="Whether or not to save the result (train and test accuuracy, loss)")

    return parser.parse_args()


LOG_INTERVAL = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parseArgs()

print("centre_and_scale:", args.centre_and_scale)
print("augment_data:", args.augment_data)

def train(seed, lr, pruning_factor):
    # Set manual seed
    torch.manual_seed(seed)

    # net
    net = get_net(args).to(device)
    
    # optimiser
    optimiser = get_optimiser(net, args)

    # Learning rate schedule
    if args.lr_schedule:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimiser, milestones=args.milestones, gamma=args.gamma)

    # Train, val, and test loaders

    train_loader, val_loader, test_loader = get_data_loaders(args, seed)

    # loss
    loss = F.cross_entropy
    
    # Initialize network
    if not (("fixup" in args.network_name) and (args.offset=='None')):        # Standard Fixup networks are initialised when the network is built, in which case this step is skipped
        for layer in net.modules():
            if (type(layer) == nn.Conv2d) or (type(layer) == nn.Linear):
                if args.init == 'normal_kaiming':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                elif args.init == 'normal_kaiming_fout':
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu', mode='fan_out')
                elif args.init == 'normal_xavier':
                    nn.init.xavier_normal_(layer.weight)
                elif args.init == 'orthogonal':
                    nn.init.orthogonal_(layer.weight)
                elif args.init == 'zeros':
                    nn.init.zeros_(layer.weight)
                else:
                    raise ValueError(f"Unrecognised initialisation parameter {args.init}")

    ############################################################################
    ####################        Pruning at init         ########################
    ############################################################################
    pruner=None
    
    if pruning_factor != 1:
        if args.rigl:
            sparsity_distributions = {3: 'uniform', 4: 'epf', 5:'epl', 6:'erk'}
            total_iterations = args.epochs*len(train_loader)
            T_end = int(0.75 * total_iterations)

            # ------------------------------------ REQUIRED LINE # 1 ------------------------------------
            # now, create the RigLScheduler object
            pruner = RigLScheduler(net,                           # model you created
                           optimiser,                       # optimizer (recommended = SGD w/ momentum)
                           dense_allocation=pruning_factor,            # a float between 0 and 1 that designates how sparse you want the network to be
                                                              # (0.1 dense_allocation = 90% sparse)
                           sparsity_distribution=sparsity_distributions[args.prune_method], # distribution hyperparam within the paper, currently only supports `uniform`
                           T_end=T_end,                     # T_end hyperparam within the paper (recommended = 75% * total_iterations)
                           delta=100,                       # delta hyperparam within the paper (recommended = 100)
                           alpha=0.3,                       # alpha hyperparam within the paper (recommended = 0.3)
                           grad_accumulation_n=1,           # new hyperparam contribution (not in the paper)
                                                              # for more information, see the `Contributions Beyond the Paper` section
                           static_topo=False,               # if True, the topology will be frozen, in other words RigL will not do it's job
                                                              # (for debugging)
                           ignore_linear_layers=False,      # if True, linear layers in the network will be kept fully dense
                           state_dict=None)                 # if you have checkpointing enabled for your training script, you should save
                                                              # `pruner.state_dict()` and when resuming pass the loaded `state_dict` into
                                                                # the pruner constructor

        else: 

            if (args.prune_method in [1]):
                if args.offset!='None':
                    raise NotImplementedError
                else: 
                    print(f'Pruning network iteratively for {args.num_steps} steps')
                    keep_masks = iterative_pruning(net, train_loader, device, pruning_factor,
                                               prune_method=args.prune_method,
                                               num_steps=args.num_steps,
                                               mode=args.mode, num_batches=args.num_batches)
                    apply_prune_mask(net, keep_masks)

            elif args.prune_method in [3,4,5,6]:

                if args.prune_method ==3: # Uniform
                    layer_densities = [[pruning_factor for _ in range(len(num_filters_per_layer))]]
                elif args.prune_method ==4: # EPF
                    layer_densities = calculate_layer_densities(net, pruning_factor, distribution_type='equal_per_filter')
                elif args.prune_method ==5: # EPL
                    layer_densities = calculate_layer_densities(net, pruning_factor, distribution_type='equal_per_layer')
                elif args.prune_method ==6: # ERK
                    layer_densities = calculate_layer_densities(net, pruning_factor, distribution_type='erk')

                keep_masks = prune(net, layer_densities, device)
                apply_prune_mask(net, keep_masks)
            
    
            print("**** Pruning complete ****")
            ds,  lay_ds = check_model_sparsity(keep_masks)
            print(f"**** Overall model density: {ds}")
            print(f"**** Model layer densities: {lay_ds}")



    if args.offset!='None':
        check_zero_init(net)

    ############################################################################
    ####################          Training              ########################
    ############################################################################
    evaluator = create_supervised_evaluator(net, {
        'accuracy': Accuracy(),
        'cross_entropy': Loss(loss)
    }, device)

    run_name = (args.network_name + '_' + args.dataset_name + f'_opt_{args.optimiser}'+f'_lr_schedule_{args.lr_schedule}' + f'_learning_rate_{lr}' +f'_decay_rate_{args.gamma}_' + '_sparsity_' +
                str(1 - pruning_factor) + f'_prune_method_{args.prune_method}' + f'_kernel_init_{args.init}' + f'_rseed_{seed}')

    # construct directory for this run
    base_dir = make_directory(args)

    if args.save_results:
        writer_name= '../runs/' + base_dir + run_name
        writer = SummaryWriter(writer_name)
        with open( '../runs/' + base_dir + run_name + '_args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        writer = None
     # Train and evaluate

    print("**** Starting training ******")
    print(f"**** {args.dataset_name} ******")
    print(f"**** {args.network_name} ******")
    print(f"**** pruning_factor: {pruning_factor} ******")
    print(f"**** pruning_method: {args.prune_method} ******")
    print(f"**** optimiser: {args.optimiser} ******")
    print(f"**** init_lr: {args.init_lr} ******")
    print(f"**** lr_schedule: {args.lr_schedule} ******")
    print(f"**** lr_decay_rate: {args.gamma} ******")

    best_acc = 0
    iterations = 0

    for epoch in range(0, args.epochs):
        if args.lr_schedule:
            lr_scheduler.step()

        train_loss = train_cross_entropy(epoch, net, train_loader, optimiser, device,
                                             writer, args, LOG_INTERVAL=20, pruner=pruner)
        iterations += len(train_loader)

        # Evaluate on validation set
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        # Save history
        avg_accuracy = metrics['accuracy']
        print(f"*** Accuracy at  epoch {epoch}: {avg_accuracy}")
        avg_cross_entropy = metrics['cross_entropy']
        if args.save_results:
            writer.add_scalar("val/loss", avg_cross_entropy, iterations)
            writer.add_scalar("val/accuracy", avg_accuracy, iterations)

        # Save copy of best_model
        is_best = avg_accuracy > best_acc
        best_acc = max(avg_accuracy, best_acc)

        if is_best:
            if args.save_model:
                if not os.path.exists(args.save_loc + base_dir):
                    os.makedirs(args.save_loc + base_dir)
                save_name = args.save_loc + base_dir + run_name + '.model'
                torch.save(net.state_dict(), save_name)
                with open(args.save_loc + base_dir + run_name + '_args.txt', 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
            best_net = copy.deepcopy(net)

        print("pruner:", pruner)

    # Evaluate final model on test set

    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"*** Test accuracy of final epoch model: {avg_accuracy}")
    avg_cross_entropy = metrics['cross_entropy']
    writer.add_scalar("test/final_epoch_loss", avg_cross_entropy, iterations)
    writer.add_scalar("test/final_epoch_accuracy", avg_accuracy, iterations)
    if args.save_model:
        save_name = args.save_loc + base_dir + run_name + 'final_epoch.model'
        torch.save(net.state_dict(), save_name)
        

    print(f"*** Max validation accuracy during training: {best_acc}")

    # Evaluate best model on test set
    test_evaluator = create_supervised_evaluator(best_net, {
        'accuracy': Accuracy(),
        'cross_entropy': Loss(loss)
    }, device)

    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"*** TEST ACCURACY OF BEST MODEL: {avg_accuracy}")
    avg_cross_entropy = metrics['cross_entropy']
    if args.save_results:
        writer.add_scalar("test/best_model_loss", avg_cross_entropy, iterations)
        writer.add_scalar("test/best_model_accuracy", avg_accuracy, iterations)


    ### Confirm sparsity is unchanged after training:

    ds, lay_ds = check_trained_model_sparsity(net)
    print(f"**** Overall model density after training: {ds}")
    print(f"**** Model layer densities after training: {lay_ds}")


if __name__ == '__main__':

    seeds = list(range(30000 * args.iterations))
    random.shuffle(seeds)
    for seed in seeds[:args.iterations]:
        train(seed, args.init_lr, args.pruning_factor)
