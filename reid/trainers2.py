from __future__ import print_function, absolute_import
import time
import sys

import torch
from torch.nn.utils import clip_grad_norm_
from .utils.meters import AverageMeter
import torch.nn as nn

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs/2)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class BaseTrainer(object):
    def __init__(self, model, discriminator, criterion, clip_value=16.0):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.clip_value = clip_value
        self.discriminator = discriminator
        self.criterion_ce = CrossEntropyLabelSmooth(self.discriminator.num_cam_ids).cuda()

    def train(self, epoch, data_loader, optimizer, optimizer_D):
        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        camera_losses_D = AverageMeter()
        camera_losses_G = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            self.model.eval()
            self.criterion.train()
            self.discriminator.train()

            data_time.update(time.time() - end)
            inputs, targets, camids = self._parse_data(inputs)

            featureD, featureT = self.model(inputs)

            predicted_ids = self.discriminator(featureD)

            if False: # 训练D
                optimizer_D.zero_grad()
                cameraid_loss_D = self.criterion_ce(predicted_ids, camids)  # optimize D
                cameraid_loss_D.backward()
                optimizer_D.step()
                camera_losses_D.update(cameraid_loss_D.item(), targets.size(0))
            else:
                # # Casts operations to mixed precision
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, acc = self.criterion(featureT, targets)
                    finite_mask = loss.isfinite()
                    if finite_mask.any():
                        loss = loss[finite_mask].mean()
                        acc = acc[finite_mask].mean()
                    else:
                        loss = acc = None

                if loss is None:
                    continue

                losses.update(loss.item(), targets.size(0))
                precisions.update(acc.item(), targets.size(0))

                # predicted_ids = torch.softmax(predicted_ids, dim=1)
                # print(predicted_ids)
                # camera_loss_G = torch.mean(torch.sum(predicted_ids * torch.log(predicted_ids) + 0.5, dim=1))
                # camera_losses_G.update(camera_loss_G.item(), targets.size(0))

                if self.clip_value > 0:
                    # Scales the loss, and calls backward() to create scaled gradients
                    scaler.scale(loss).backward()
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                clip_grad_norm_(self.model.parameters(), self.clip_value)
                clip_grad_norm_(self.criterion.parameters(), self.clip_value)

                if self.clip_value > 0:
                    # Unscales gradients and calls or skips optimizer.step()
                    scaler.step(optimizer)
                    # Updates the scale for next iteration
                    scaler.update()
                else:
                    optimizer.step()

                batch_time.update(time.time() - end)
                end = time.time()

            print('Epoch: [{}][{}/{}]. '
                  'Time: {:.3f} ({:.3f}). '
                  'Data: {:.3f} ({:.3f}). '
                  'Loss: {:.3f} ({:.3f}). '
                  'Prec: {:.2%} ({:.2%}).'
                  'g_loss: {:.2f} ({:.2f}). '
                  'd_rec: {:.2f} ({:.2f}).'
                  .format(epoch + 1, i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          losses.val, losses.avg,
                          precisions.val, precisions.avg,
                          camera_losses_G.val, camera_losses_G.avg,
                          camera_losses_D.val, camera_losses_D.avg), end='\r', file=sys.stdout.console)
            
        return losses.avg, precisions.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, camids = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        camids = camids.cuda()
        return inputs, targets, camids

    def _forward(self, inputs, targets):
        feature = self.model(inputs)
        #print(feature.shape)
        loss, acc = self.criterion(feature, targets)
        finite_mask = loss.isfinite()
        if finite_mask.any():
            loss = loss[finite_mask].mean()
            acc = acc[finite_mask].mean()
        else:
            loss = acc = None
        return loss, acc
