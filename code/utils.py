import numpy as np
import torch.nn as nn
import torch.optim as optim
from code.models import *
from code.datasets import *

dataset_num_classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'tiny_imagenet': 200
}

network_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet110': resnet110,
    'lenet': lenet,
    'mobilenet': mobilenet,
    'fixup_resnet110': fixup_resnet110,
    'vgg19': vgg19,
}

def get_net(args):
    """
    note the importance of order in these conditionals (e.g. lenet is contained in mobilenet, etc)
    """
    print(args.alpha_trainable)
    if 'fixup_resnet' in args.network_name:
        return network_dict[args.network_name](num_classes=dataset_num_classes[args.dataset_name], offset=args.offset, alpha_trainable=args.alpha_trainable)
    elif 'vgg' in args.network_name:
        return network_dict[args.network_name](dataset=args.dataset_name, offset=args.offset, alpha_trainable=args.alpha_trainable)
    elif 'resnet' in args.network_name:
        return network_dict[args.network_name](num_classes=dataset_num_classes[args.dataset_name], stable_resnet=args.stable_resnet, in_planes=args.in_planes, offset=args.offset, alpha_trainable=args.alpha_trainable)
    elif 'mobilenet' in args.network_name:
        return network_dict[args.network_name](num_classes=dataset_num_classes[args.dataset_name], offset=args.offset, alpha_trainable=args.alpha_trainable)
    elif 'lenet' in args.network_name:
        return network_dict[args.network_name](offset=args.offset, alpha_trainable=args.alpha_trainable)
        

def get_optimiser(net, args):
    if 'fixup_resnet' in args.network_name:
        parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
        parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
        parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
        if args.optimiser == "sgd":
            optimiser = optim.SGD([{'params': parameters_bias, 'lr': args.init_lr/10.},
                                    {'params': parameters_scale, 'lr': args.init_lr/10.},
                                    {'params': parameters_others}],
                                    lr=args.init_lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

        elif args.optimiser == "adam":
            optimiser = optim.Adam([{'params': parameters_bias, 'lr': args.init_lr/10.},
                                    {'params': parameters_scale, 'lr': args.init_lr/10.},
                                    {'params': parameters_others}],
                                   lr=args.init_lr,
                                   weight_decay=args.weight_decay)
        
    else:
        if args.optimiser == 'sgd':
            optimiser = optim.SGD(net.parameters(), lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay)    
        elif args.optimiser == 'adam':
            optimiser = optim.Adam(net.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

    return optimiser
            
    
def get_data_loaders(args, seed):
    if 'CIFAR' in args.dataset_name:
        train_loader, val_loader = get_cifar_train_valid_loader(
                    batch_size=args.batch_size,
                    augment= args.augment_data,
                    random_seed=seed,
                    valid_size=1-args.frac_data_for_train,
                    pin_memory=False,
                    dataset_name=args.dataset_name,
                    centre_and_scale = args.centre_and_scale
                    )

        test_loader = get_cifar_test_loader(
                    batch_size=args.test_batch_size,
                    pin_memory=False,
                    dataset_name=args.dataset_name,
                    centre_and_scale=args.centre_and_scale
                    )

    elif args.dataset_name == 'tiny_imagenet':
        train_loader, val_loader, test_loader = get_tiny_imagenet_train_valid_loader(args.batch_size,
                                                                                     augment=args.augment_data,
                                                                                     shuffle=True,
                                                                                     num_workers=args.num_workers)
                                                                             
    return train_loader, val_loader, test_loader
                                                                                                                                                                                                                       
    
def make_directory(args):

    base_dir = args.dataset_name + '/'
    if args.augment_data:
        base_dir += 'data_augmentation_true/'
    else:
        base_dir += 'data_augmentation_false/'

    base_dir += 'centre_and_scale_true/'

    base_dir += args.network_name + '/'

    base_dir+= args.optimiser + '/'

    if args.offset == 'dct':
        base_dir += 'dct_plus_sparse/'
    else:
        base_dir+='no_offset/'
                                          
    if args.rigl:
        base_dir+= 'rigl/'
    else:
        base_dir += 'static_sparse/'

    if args.alpha_trainable==False:
        base_dir += 'alpha_fixed/'
        
    if args.prune_method == 1:
        base_dir += 'force/'
    elif args.prune_method ==3:
        base_dir += 'uniform_random/'
    elif args.prune_method ==4:
        base_dir += 'equal_per_filter/'
    elif args.prune_method ==5:
        base_dir += 'equal_per_layer/'
    elif args.prune_method == 6:
        base_dir += 'erk/'
    elif args.prune_method == 7:
        base_dir += 'IMP/'

    base_dir += str(args.pruning_factor) + '/' 
    return base_dir   

def calculate_layer_densities(net, target_density, distribution_type = "equal_per_filter", boost_factor = 10):
    """
    **inputs**
    target_dinsity: global network connection density
    distribution_type: method for how to allocate non_zeros between layers

    **output**
    a list of layer densities

    Notes:
    - all above excl bias - which are not sparsified, if used
    - equal per layer assigns any remainig available non-zeros to the final layer
    """
    param_dim_sums = []
    full_params = []
    num_filters_per_layer = []
    for layer in net.modules():
        if type(layer)==(nn.Conv2d) or type(layer)==(nn.Linear):
            full_params.append(np.product(layer.weight.shape))
            num_filters_per_layer.append(layer.weight.shape[0])
            param_dim_sums.append(np.sum(layer.weight.shape))

    target_non_zeros = target_density*sum(full_params)
   
    if distribution_type == "equal_per_filter":
        num_filters_total = sum(num_filters_per_layer)
        non_zeros_per_filter = target_non_zeros/num_filters_total
        non_zeros_per_layer = [non_zeros_per_filter*filters for filters in num_filters_per_layer]
        layer_densities = np.minimum(1, np.divide(non_zeros_per_layer, full_params))

    elif distribution_type == "equal_per_filter_boost_last_layer":
        num_filters_total = sum(num_filters_per_layer)
        non_zeros_per_filter = target_non_zeros/num_filters_total
        non_zeros_per_layer = [non_zeros_per_filter*filters for filters in num_filters_per_layer]
        non_zeros_per_layer[-1] = non_zeros_per_layer[-1]*boost_factor
        layer_densities = np.minimum(1, np.divide(non_zeros_per_layer, full_params))

    elif distribution_type == "equal_per_layer":
        non_zeros_per_layer = [target_non_zeros/len(full_params) for _ in range(len(full_params))]
        non_zeros_per_layer = [np.minimum(full_params[i], non_zeros_per_layer[i]) for i in range(len(full_params))]
        remaining_non_zeros = target_non_zeros - sum(non_zeros_per_layer)
        non_zeros_per_layer[-1]+=remaining_non_zeros
        layer_densities = np.minimum(np.divide(non_zeros_per_layer, full_params), 1)

    elif distribution_type == "erk":
        denom = np.sum(param_dim_sums)
        total = np.sum(full_params)
        epsilon = target_density * total / denom
        layer_densities = np.minimum(np.array([epsilon*(param_dim_sums[i])/(full_params[i]) for i in range(len(full_params))]), 1)

    elif distribution_type == "uniform":
        layer_densities = np.array([target_density for _ in range(len(num_filters_per_layer))])

    return layer_densities


def check_model_sparsity(masks):
    total_param_count = 0
    total_nonzero_count = 0
    layer_densities = {}

    for i, mask in enumerate(masks):
        # m = mask.cpu().numpy()
        # dim = np.product(tuple(m.shape))
        dim = mask.numel()
        nz=torch.nonzero(mask).size()[0]
        layer_densities[i] = nz/dim
        total_param_count+= dim
        total_nonzero_count += nz

    return total_nonzero_count/total_param_count, layer_densities

def check_trained_model_sparsity(net):
    total_param_count = 0
    total_nonzero_count = 0
    layer_densities = {}
    for i, layer in enumerate(net.modules()):
        if (type(layer) == nn.Conv2d) or (type(layer) == nn.Linear):
            dim = layer.weight.numel()
            # print("dim", dim)
            nz = torch.nonzero(layer.weight).size()[0]
            # print("nz", nz)
            layer_densities[i] = nz/dim
            total_param_count+= dim
            total_nonzero_count += nz

    return total_nonzero_count/total_param_count, layer_densities

def check_zero_init(net):
    non_zero_weights_count = 0
    for layer in net.modules():
        if (type(layer) == nn.Conv2d) or (type(layer) == nn.Linear):
            if torch.nonzero(layer.weight).size()[0] > 0:
                print("*** Weight not initialised to 0! ***")
                non_zero_weights_count+=1
    if non_zero_weights_count==0:
        print("*** All weights initialised to 0 ***")

    return None


def train_cross_entropy(epoch, model, train_loader, optimizer, device, writer, args, LOG_INTERVAL=20, pruner = None):
    '''
    Util method for training with cross entropy loss.
    '''
    # Signalling the model that it is in training mode
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        # Loading the data onto the GPU
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(data)

        loss = F.cross_entropy(logits, labels)

        loss.backward()

        train_loss += loss.item()

        if args.rigl:
            if pruner():    
                optimizer.step()

        else: 
            optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader) * len(data),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.save_results:
                writer.add_scalar("training/loss", loss.item(),
                              epoch*len(train_loader)+batch_idx)
        
        
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))
    return train_loss / len(train_loader)


