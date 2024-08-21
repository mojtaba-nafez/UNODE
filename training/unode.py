import time

import torch.optim

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from datasets.negative_pair_generation import NegativePairGenerator
from utils.utils import AverageMeter, normalize
import numpy as np

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    neg_pair_gen = NegativePairGenerator(P)

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()
    losses['shift'] = AverageMeter()

    check = time.time()
    
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n  # number of trained samples
        data_time.update(time.time() - check)
        check = time.time()
        
        ### SimCLR loss ###
        if P.dataset != 'imagenet':
            batch_size = images.size(0)
            images = images.to(device)
            if P.no_hflip:
                images1, images2 = images.repeat(2, 1, 1, 1).chunk(2)  # hflip
            else:
                images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2)  # hflip
        else:
            batch_size = images[0].size(0)
            images1, images2 = images[0].to(device), images[1].to(device)
        labels = labels.to(device)
        images1 = torch.cat([images1.clone(), neg_pair_gen.create_negative_pair(images1.clone())])
        images2 = torch.cat([images2.clone(), neg_pair_gen.create_negative_pair(images2.clone())])
      
        shift_labels = torch.cat([torch.ones_like(labels), torch.zeros_like(labels)], 0)  # B -> 4B
        shift_labels = shift_labels.repeat(2)
        
        images_pair = torch.cat([images1, images2], dim=0)  # 8B
        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=False, shift=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss_sim = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda

        loss_shift = criterion(outputs_aux['shift'], shift_labels)

        ### total loss ###
        loss = loss_sim + loss_shift

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        losses['cls'].update(0, batch_size)
        losses['sim'].update(loss_sim.item(), batch_size)
        losses['shift'].update(loss_shift.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f] [LossSim %f] [LossShift %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, losses['sim'].value, losses['shift'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f] [LossSim %f] [LossShift %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average, losses['sim'].average, losses['shift'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/loss_shift', losses['shift'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

