import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.transform_layers as TL
from utils.utils import set_random_seed, normalize, get_auroc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def eval_ood_detection(P, model, id_loader, ood_loaders, train_loader=None, simclr_aug=None):
    auroc_dict = dict()

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory
    prefix = os.path.join(base_path, f'feats_{P.ood_samples}_resize_factor_{P.resize_factor}')

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': ('simclr', 'shift'),
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    P.axis = []
    axis = feats_train['simclr'].mean(dim=1)
    P.axis.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))
 
    sim_norm = feats_train['simclr'].mean(dim=1).norm(dim=1)
    shi_mean = feats_train['shift'].mean(dim=1)[:, 0]
    P.weight_sim = 1 / sim_norm.mean().item()
    P.weight_shi = 1 / shi_mean.mean().item()
    
    print(f'weight_sim: {P.weight_sim}')
    print(f'weight_shi: {P.weight_shi}')

    print('Pre-compute features...')
    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)

    print(f'Compute OOD scores...')
    scores_id = get_scores(P, feats_id).numpy()
    scores_ood = dict()
    if P.normal_class is not None:
        one_class_score = []

    for ood, feats in feats_ood.items():
        scores_ood[ood] = get_scores(P, feats).numpy()
        auroc_dict[ood]= get_auroc(scores_id, scores_ood[ood])
        if P.normal_class is not None:
            one_class_score.append(scores_ood[ood])

    if P.normal_class is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict


def get_scores(P, feats_dict):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = f_sim.mean(dim=0, keepdim=True)  # list of (1, d)
        f_shi = f_shi.mean(dim=0, keepdim=True)  # list of (1, 4)
        score = 0
        score += (f_sim * P.axis[0]).sum(dim=1).max().item() * P.weight_sim
        score += f_shi[:, 0].item() * P.weight_shi
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(P, data_name, model, loader, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    feats_dict = dict()
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats 

    return feats_dict


def _get_features(P, model, loader, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x
        x = x.to(device)  # gpu tensor
        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)
            x_t = simclr_aug(x)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(1)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, 1, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

