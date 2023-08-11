from __future__ import print_function, absolute_import
import time
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, clip_value=16.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.clip_value = clip_value
        self.newmodel = None
        self.newcriterion = None


    def train(self, args, epoch, data_loader1, data_loader2, optimizer):
        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        losses_meta = AverageMeter()
        precisions_meta = AverageMeter()

        end = time.time()
        for i, (inputs, inputs2) in enumerate(zip(data_loader1,data_loader2)):
            self.model.eval()
            self.criterion.train()

            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)
            optimizer.zero_grad()
            # Casts operations to mixed precision

            loss, acc = self._forward(inputs, targets)
            if loss is None: continue

            losses.update(loss.item(), targets.size(0))
            precisions.update(acc.item(), targets.size(0))

            loss.backward()
            # clip_grad_norm_(self.model.parameters(), self.clip_value)
            # clip_grad_norm_(self.criterion.parameters(), self.clip_value)
            optimizer.first_step(zero_grad=True)

            inputs2, targets2 = self._parse_data(inputs2)
            loss2, acc2 = self._forward(inputs2, targets2)
            if loss2 is None: continue
            losses_meta.update(loss2.item(), targets.size(0))
            precisions_meta.update(acc2.item(), targets.size(0))

            loss2.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_value)
            clip_grad_norm_(self.criterion.parameters(), self.clip_value)
            optimizer.second_step(zero_grad=True)

            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{}][{}/{}]. '
                  'Time: {:.3f} ({:.3f}). '
                  'Data: {:.3f} ({:.3f}). '
                  'Loss: {:.3f} ({:.3f}). '
                  'Prec: {:.2%} ({:.2%}).'
                  'Loss_Meta: {:.3f} ({:.3f}). '
                  'Prec_Meta: {:.2%} ({:.2%}).'
                  .format(epoch + 1, i + 1, len(data_loader1),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses.val, losses.avg,
                          precisions.val, precisions.avg,
                          losses_meta.val, losses_meta.avg,
                          precisions_meta.val, precisions_meta.avg), end='\r', file=sys.stdout.console)

        return losses.avg, precisions.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

    def _forward2(self, inputs, targets):
        raise NotImplementedError
#
class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, inputs, targets):
        _, feature = self.model(inputs)
        # rint(feature.shape)
        loss, acc = self.criterion(feature, targets)
        finite_mask = loss.isfinite()
        if finite_mask.any():

            loss = loss[finite_mask].mean()
            acc = acc[finite_mask].mean()
        else:
            loss = acc = None
        return loss, acc

    def _forward2(self, inputs, targets):
        _, feature = self.newmodel(inputs)
        # print(feature.shape)
        loss, acc = self.newcriterion(feature, targets)
        finite_mask = loss.isfinite()
        if finite_mask.any():
            loss = loss[finite_mask].mean()
            acc = acc[finite_mask].mean()
        else:
            loss = acc = None
        return loss, acc
