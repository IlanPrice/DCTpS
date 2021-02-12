import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from code.pruning_algos import prune, iterative_pruning
from code.experiments import *
from code.mask_networks import apply_prune_mask
from code.utils import calculate_layer_densities, check_model_sparsity, check_trained_model_sparsity, check_zero_init

import os
import argparse
import random

import shutil
import copy

def parseArgs():

    parser = argparse.ArgumentParser(
                description="Training CIFAR / Tiny-Imagenet.",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pruning_factor", type=float, nargs = "*", default=[0.01], dest="pruning_factor",
                        help='Fraction of connections after pruning')

    parser.add_argument("--prune_method", type=int, default=5, dest="prune_method",
                        help="""Which pruning method to use:
                                1->FORCE
                                3->Uniform Random
                                4->equal_per_filter
                                5->equal_per_layer""")

    parser.add_argument("--dataset", type=str, default='CIFAR10',
                        dest="dataset_name", help='Dataset to train on')

    parser.add_argument("--network_name", type=str, default='resnet50', dest="network_name",
                        help='Model to train')

    parser.add_argument("--num_steps", type=int, default=10,
                        help='Number of steps to use with iterative pruning')

    parser.add_argument("--mode", type=str, default='exp',
                        help='Mode of creating the iterative pruning steps one of "linear" or "exp".')

    parser.add_argument("--num_batches", type=int, default=1,
                        help='''Number of batches to be used during FORCE pruning''')

    parser.add_argument("--save_loc", type=str, default='../saved_models/',
                        dest="save_loc", help='Path where to save the model')

    parser.add_argument("--opt", type=str, default='sgd',
                        dest="optimiser",
                        help='Choice of optimisation algorithm - options are SGD and adam')

    parser.add_argument("--init_lr", type=float, nargs = "*", default = [0.1],
                        dest="init_lr",
                        help='Choice of initial learning rate')

    parser.add_argument("--lr_schedule", action = 'store_true', default = False,
                        help="Indicates that lr should decrease by the speacified factor gamma")

    parser.add_argument("--frac-train-data", type=float, default=0.9, dest="frac_data_for_train",
                        help='Fraction of data used for training')

    parser.add_argument("--init", type=str, default='normal_kaiming',
                        help='Which initialization method to use for the trainable weight tensors')

    parser.add_argument("--in_planes", type=int, default=64,
                        help='''Number of input planes in Resnet. Afterwards they duplicate after
                        each conv with stride 2 as usual.''')

    parser.add_argument("--dont_augment_data", action = 'store_true', default = False,
                        help="Assert that data is not augmented with random flips, etc")

    parser.add_argument("--dont_centre_and_scale", action = 'store_true', default = False,
                        help="Assert that data is not normalised")

    parser.add_argument("--iterations", type=int, default=1, dest="iterations",
                    help="How many iteratinons to run per combination of hyperparameters")

    parser.add_argument("--lr_decay_rate", type=float, default = 0.1,
                        dest="gamma",
                        help = 'Factor multiplied by learning rate at specified epoch milestones if --lr_schedule is True')

    return parser.parse_args()


LOG_INTERVAL = 20
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parseArgs()

augment_data = not args.dont_augment_data
centre_and_scale = not args.dont_centre_and_scale

print("centre_and_scale:", centre_and_scale)
print("augment_data:", augment_data)

def train(seed, lr, pruning_factor):

    # Set manual seed
    torch.manual_seed(seed)

    if 'fixup' in args.network_name:
        [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = fixupresnet_cifar_experiment(device, args.network_name,
                                                                   args.dataset_name, args.optimiser,
                                                                  args.frac_data_for_train, lr, args.gamma, augment_data, centre_and_scale, seed)

    elif 'resnet' in args.network_name:
        stable_resnet = False
        if 'stable' in args.network_name:
            stable_resnet = True
        if 'CIFAR' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = resnet_cifar_experiment(device, args.network_name,
                                                                   args.dataset_name, args.optimiser,
                                                                   args.frac_data_for_train, lr, args.gamma,
                                                                   stable_resnet, args.in_planes, augment_data, centre_and_scale, seed)
        elif 'tiny_imagenet' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = resnet_tiny_imagenet_experiment(device, args.network_name,
                                                                          args.dataset_name, args.in_planes, args.optimiser, lr, args.gamma, seed)


    elif 'vgg' in args.network_name or 'VGG' in args.network_name:
        if 'tiny_imagenet' in args.dataset_name:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = vgg_tiny_imagenet_experiment(device, args.network_name,
                                                                       args.dataset_name)
        else:
            [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = vgg_cifar_experiment(device, args.network_name,
                                                               args.dataset_name, args.optimiser, args.frac_data_for_train, lr, args.gamma,
                                                               augment_data, centre_and_scale, seed)

    elif 'mobilenet' in args.network_name:
        [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = mobilenet_cifar_experiment(device, args.network_name,
                                                                   args.dataset_name, args.optimiser,
                                                                   args.frac_data_for_train, lr, args.gamma, augment_data, centre_and_scale, seed)

    elif 'lenet' in args.network_name:
        [net, optimiser, lr_scheduler,
             train_loader, val_loader,
             test_loader, loss, EPOCHS] = lenet_cifar_experiment(device, args.network_name,
                                                               args.dataset_name, args.optimiser, args.frac_data_for_train, lr, augment_data, centre_and_scale, seed)



    # Initialize network
    if ("fixup" not in args.network_name) or ("dctplussparse" in args.network_name):        # Fixup (non-DCTpS) networks are initialised when the network is built, in which case this step is skipped

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

    if args.prune_method in [1]:
        #Prune method 1 = FORCE
        if 'dctplussparse' in args.network_name:
            if pruning_factor != 1:
                raise NotImplementedError
        else:
            if pruning_factor != 1:
                print(f'Pruning network iteratively for {args.num_steps} steps')
                keep_masks = iterative_pruning(net, train_loader, device, pruning_factor,
                                               prune_method=args.prune_method,
                                               num_steps=args.num_steps,
                                               mode=args.mode, num_batches=args.num_batches)
                apply_prune_mask(net, keep_masks)

    elif args.prune_method in [3,4,5]:

        if args.prune_method ==3: # Uniform
            densities_options = [[pruning_factor for _ in range(len(num_filters_per_layer))]]
        elif args.prune_method ==4: # EPF
            densities_options = calculate_layer_densities(net, [pruning_factor], distribution_type='equal_per_filter')
        elif args.prune_method ==5: # EPL
            densities_options = calculate_layer_densities(net, [pruning_factor], distribution_type='equal_per_layer')

        if pruning_factor != 1:
            keep_masks = prune(net, densities_options[0], device)
            apply_prune_mask(net, keep_masks)


    if pruning_factor != 1:
        print("**** Pruning complete ****")
        ds,  lay_ds = check_model_sparsity(keep_masks)
        print(f"**** Overall model density: {ds}")
        print(f"**** Model layer densities: {lay_ds}")

    if 'dctplussparse' in args.network_name:
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
    base_dir = args.dataset_name + '/'
    if augment_data:
        base_dir += 'data_augmentation_true/'
    else:
        base_dir += 'data_augmentation_false/'

    base_dir += 'centre_and_scale_true/'

    base_dir += args.network_name + '/'

    base_dir+= args.optimiser + '/'

    if args.prune_method in [3,4,5]:
        if 'dctplussparse' in args.network_name:
            base_dir += 'dct_plus_sparse/'
        else:
            base_dir += 'random_sparse/'
    elif args.prune_method == 1:
        if 'dctplussparse' in args.network_name:
            base_dir += 'dct_plus_sparse/'
        base_dir += 'force/'

    if args.prune_method ==3:
        base_dir += 'uniform_random/'
    elif args.prune_method ==4:
        base_dir += 'equal_per_filter/'
    elif args.prune_method ==5:
        base_dir += 'equal_per_layer/'

    base_dir += str(pruning_factor) + '/'

    writer_name= '../runs/' + base_dir + run_name
    writer = SummaryWriter(writer_name)



     # Train and evaluate

    iterations = 0
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

    for epoch in range(0, EPOCHS):
        if args.lr_schedule:
            lr_scheduler.step()

        train_loss = train_cross_entropy(epoch, net, train_loader, optimiser, device,
                                             writer, LOG_INTERVAL=20)
        iterations +=len(train_loader)

        # Evaluate on validation set
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        # Save history
        avg_accuracy = metrics['accuracy']
        print(f"*** Accuracy at  epoch {epoch}: {avg_accuracy}")
        avg_cross_entropy = metrics['cross_entropy']
        writer.add_scalar("val/loss", avg_cross_entropy, iterations)
        writer.add_scalar("val/accuracy", avg_accuracy, iterations)

        # Save copy of best_model
        is_best = avg_accuracy > best_acc
        best_acc = max(avg_accuracy, best_acc)

        if is_best:
            if not os.path.exists(args.save_loc + base_dir):
                os.makedirs(args.save_loc + base_dir)
            save_name = args.save_loc + base_dir + run_name + '.model'
            torch.save(net.state_dict(), save_name)
            best_net = copy.deepcopy(net)


    # Evaluate final model on test set

    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    print(f"*** Test accuracy of final epoch model: {avg_accuracy}")
    avg_cross_entropy = metrics['cross_entropy']
    writer.add_scalar("test/final_epoch_loss", avg_cross_entropy, iterations)
    writer.add_scalar("test/final_epoch_accuracy", avg_accuracy, iterations)
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
    writer.add_scalar("test/best_model_loss", avg_cross_entropy, iterations)
    writer.add_scalar("test/best_model_accuracy", avg_accuracy, iterations)


    ### Confirm sparsity is unchanged after training:

    # ds, lay_ds = check_trained_model_sparsity(net)
    # print(f"**** Overall model density after training: {ds}")
    # print(f"**** Model layer densities after training: {lay_ds}")


if __name__ == '__main__':
    for pf in args.pruning_factor:
        for l in args.init_lr:
            seeds = list(range(30000 * args.iterations))
            random.shuffle(seeds)
            for seed in seeds[:args.iterations]:
                train(seed, l, pf)
