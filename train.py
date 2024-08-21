import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from utils.utils import (
    Logger, save_checkpoint, save_linear_checkpoint,
    load_checkpoint, get_loader_unique_label, count_parameters
)
from args import parse_args
import models.classifier as C
from datasets import (
    set_dataset_count, get_dataset, get_superclass_list, get_subclass_dataset
)
from training.unode import train
from training.scheduler import GradualWarmupScheduler

def initialize():
    P = parse_args()
    if P.dataset=='mnist' or P.dataset=='svhn-10':
        P.no_hflip = True
    cls_list = get_superclass_list(P.dataset)
    anomaly_labels = [elem for elem in cls_list if elem not in [P.normal_class]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P.multi_gpu = False
    return P, anomaly_labels, device

def prepare_datasets(P):
    train_set, test_set, image_size, n_classes = get_dataset(
        P, dataset=P.dataset, download=True, image_size=(P.image_size, P.image_size, 3), labels=[P.normal_class]
    )
    P.image_size = image_size
    P.n_classes = n_classes

    full_test_set = deepcopy(test_set)  # test set of full classes

    if P.dataset in ['ISIC2018', 'mvtecad', 'cifar10-versus-100', 'cifar100-versus-10']:
        train_set = set_dataset_count(train_set, count=P.normal_data_count)
        test_set = get_subclass_dataset(P, test_set, classes=[0])
    else:
        train_set = get_subclass_dataset(P, train_set, classes=[P.normal_class], count=P.normal_data_count)
        test_set = get_subclass_dataset(P, test_set, classes=[P.normal_class])

    return train_set, test_set, full_test_set

def prepare_dataloaders(train_set, test_set, P):
    kwargs = {'pin_memory': False, 'num_workers': 4}
    train_loader = DataLoader(train_set, shuffle=True, batch_size=P.batch_size, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    return train_loader, test_loader

def prepare_ood_loaders(anomaly_labels, full_test_set, P, kwargs):
    ood_test_loader = {}
    for ood in anomaly_labels:
        ood_test_set = get_subclass_dataset(P, full_test_set, classes=ood)
        ood_label = f'one_class_{ood}'
        ood_test_loader[ood_label] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)
    return ood_test_loader

def prepare_model_and_optim(P, device):
    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
    model = C.get_classifier(
        P.model, n_classes=P.n_classes, activation=P.activation_function, freezing_layer=P.freezing_layer
    ).to(device)
    model = C.get_shift_classifer(model, 2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    if P.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
        lr_decay_gamma = 0.1
    elif P.optimizer == 'lars':
        from torchlars import LARS
        base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
        optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
        lr_decay_gamma = 0.1
    else:
        raise NotImplementedError()

    if P.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
    elif P.lr_scheduler == 'step_decay':
        milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    else:
        raise NotImplementedError()

    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

    return model, simclr_aug, criterion, optimizer, scheduler_warmup

def resume_training(P, model, optimizer):
    if P.resume_path is not None:
        model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
        model.load_state_dict(model_state, strict=not P.no_strict)
        optimizer.load_state_dict(optim_state)
        start_epoch = config['epoch']
        resume = True
    else:
        start_epoch = 1
        resume = False
    return resume, start_epoch

def initialize_logger(P, resume):
    fname = f'unode_{P.dataset}_{P.model}'
    if P.normal_class is not None:
        fname += f'_one_class_{P.normal_class}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    logger = Logger(fname, ask=not resume)
    logger.log(P)
    return logger

def train_model(P, model, criterion, optimizer, scheduler_warmup, train_loader, logger, start_epoch, linear, linear_optim, simclr_aug):
    start_time = time.time()
    for epoch in range(start_epoch, P.epochs + 1):
        if P.timer is not None and P.timer < (time.time() - start_time):
            break
        logger.log_dirname(f"Epoch {epoch}")
        model.train()

        if P.multi_gpu:
            train_sampler.set_epoch(epoch)

        kwargs = {'linear': linear, 'linear_optim': linear_optim, 'simclr_aug': simclr_aug}

        if epoch > P.unfreeze_pretrain_model_epoch:
            for param in model.parameters():
                param.requires_grad = True

        train(P, epoch, model, criterion, optimizer, scheduler_warmup, train_loader, logger=logger, **kwargs)
        model.eval()
        save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)

        if (epoch % P.eval_steps == 0):
            evaluate_model(P, logger)

    epoch += 1
    save_states = model.module.state_dict() if P.multi_gpu else model.state_dict()
    save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
    save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

def evaluate_model(P, logger):
    torch.cuda.empty_cache()
    from evals.evaluation import eval_ood_detection
    P.load_path = logger.logdir + '/last.model'
    import subprocess

    arguments_to_pass = [
        "--image_size", str(P.image_size[0]),
        "--dataset", str(P.dataset),
        "--model", str(P.model),
        "--print_score",
        "--resize_fix",
        "--ood_samples", "10",
        "--resize_factor", str(0.54),
        "--load_path", str(P.load_path),
        "--normal_class", str(P.normal_class),
        '--activation_function', str(P.activation_function)
    ]

    result = subprocess.run(["python", "eval.py"] + arguments_to_pass, capture_output=True, text=True)

    if result.returncode == 0:
        logger.log("Script executed successfully.")
        logger.log("Output:")
        logger.log(result.stdout)
    else:
        logger.log("Script execution failed.")
        logger.log("Error:")
        logger.log(result.stderr)

def main():
    P, anomaly_labels, device = initialize()
    print("anomaly_labels: ", anomaly_labels)
    train_set, test_set, full_test_set = prepare_datasets(P)
    train_loader, test_loader = prepare_dataloaders(train_set, test_set, P)
    
    print("len train_set", len(train_set))
    print("len test_set", len(test_set))
    print("Unique labels(test_loader):", get_loader_unique_label(test_loader))
    print("Unique labels(train_loader):", get_loader_unique_label(train_loader))
    
    ood_test_loader = prepare_ood_loaders(anomaly_labels, full_test_set, P, {'pin_memory': False, 'num_workers': 4})

    model, simclr_aug, criterion, optimizer, scheduler_warmup = prepare_model_and_optim(P, device)
    resume, start_epoch = resume_training(P, model, optimizer)
    
    count_parameters(model)
    logger = initialize_logger(P, resume)
    logger.log(model)

    linear = model.module.linear if P.multi_gpu else model.linear
    linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

    train_model(P, model, criterion, optimizer, scheduler_warmup, train_loader, logger, start_epoch, linear, linear_optim, simclr_aug)

if __name__ == '__main__':
    main()
