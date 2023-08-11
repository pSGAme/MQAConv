from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import string
import time
import json
import random

import torch
from torch.backends import cudnn
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import StepLR

from reid import datasets
from reid.models import restranmap
from reid.models.transmatcher import TransMatcher
from reid.models import resmap
from reid.models.qaconv import QAConv
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint

from reid.utils.data.graph_sampler import GraphSampler
from reid.loss.pairwise_matching_loss import PairwiseMatchingLoss
from reid.loss.triplet_loss import TripletLoss

import sys

def get_data(dataname, data_dir, model, matcher, save_path, args):
    root = osp.join(data_dir, dataname)
    dataset = datasets.create(dataname, root, combine_all=args.combine_all)
    num_classes = dataset.num_train_ids
    train_transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=InterpolationMode.BICUBIC),
        T.Pad(10),
        T.RandomCrop((args.height, args.width)),
        T.RandomHorizontalFlip(0.5),
        T.RandomRotation(5), 
        T.ColorJitter(brightness=(0.5, 2.0), contrast=(0.5, 2.0), saturation=(0.5, 2.0), hue=(-0.1, 0.1)),
        T.RandomOcclusion(args.min_size, args.max_size),
        T.ToTensor(),
    ])

    test_transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

    train_path = osp.join(dataset.images_dir, dataset.train_path)
    train_loader = DataLoader(
        Preprocessor(dataset.train, root=train_path, transform=train_transformer),
        batch_size=args.batch_size, num_workers=args.workers,
        sampler=GraphSampler(dataset.train, train_path, test_transformer, model, matcher, args.batch_size,
                             args.num_instance,
                             args.test_gal_batch, args.test_prob_batch, save_path, args.gs_verbose),
        pin_memory=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=args.test_fea_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=args.test_fea_batch, num_workers=args.workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, query_loader, gallery_loader


def get_test_data(dataname, data_dir, height, width, workers=8, test_batch=64):
    root = osp.join(data_dir, dataname)

    dataset = datasets.create(dataname, root, combine_all=False)

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
    ])

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.images_dir, dataset.query_path), transform=test_transformer),
        batch_size=test_batch, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=test_batch, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, query_loader, gallery_loader


def main(args):
    cudnn.deterministic = False
    cudnn.benchmark = True # faster and less reproducible

    # if args.seed is not None:
    #     # reproducibility
    #     # you should set args.seed to None in real application :)
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)
    #     cudnn.deterministic = True # slower and more reproducible
    #     cudnn.benchmark = False


    exp_database_dir = osp.join(args.exp_dir, string.capwords(args.dataset))
    output_dir = osp.join(exp_database_dir, args.method, args.sub_method)
    log_file = osp.join(output_dir, 'log_x.txt')
    # Redirect print to both console and log file
    sys.stdout = Logger(log_file)

    # Create model
    ibn_type = args.ibn
    if ibn_type == 'none':
        ibn_type = None
    model = restranmap.create(args.arch, ibn_type=ibn_type, final_layer=args.final_layer, neck=args.neck,
                              nhead=args.nhead,  num_encoder_layers=args.num_trans_layers,
                              dim_feedforward=args.dim_feedforward, scale_sizes=args.scale_sizes,
                              neck2 = args.multi_scale_neck).cuda()

    num_features = model.num_features # channel, which equals to neck=512

    feamap_factor = {'layer2': 8, 'layer3': 16, 'layer4': 32}
    hei = args.height // feamap_factor[args.final_layer]
    wid = args.width // feamap_factor[args.final_layer]  # 24 * 8

    split_layer = [args.neck]
    if args.scale_sizes!="":
        split_layer.extend( [args.multi_scale_neck] * len(args.scale_sizes.split(",")) )
    if args.use_transformer:
        split_layer.extend([args.neck])

    matcher = TransMatcher(hei * wid, num_features, split_layer, args.dim_feedforward,
                           use_transformer=args.use_transformer).cuda()
    for arg in sys.argv:
        print('%s ' % arg, end='')
    print('\n')

    # Criterion
    # if args.QAConv:
    #     criterion = TripletLoss(matcher, args.margin).cuda()
    # else:
    #     criterion = PairwiseMatchingLoss(matcher).cuda()
    criterion = PairwiseMatchingLoss(matcher).cuda()
    # Optimizer
    base_param_ids = set(map(id, model.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    param_groups = [
        {'params': model.base.parameters(), 'lr': 0.1 * args.lr},
        {'params': new_params, 'lr': args.lr},
        {'params': matcher.parameters(), 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Load from checkpoint
    start_epoch = 0

    if args.resume or args.evaluate:
        print('Loading checkpoint...')
        if args.resume and (args.resume != 'ori'):
            checkpoint = load_checkpoint(args.resume)
        else:
            checkpoint = load_checkpoint(osp.join(output_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['model'])
        criterion.load_state_dict(checkpoint['criterion'])
        optimizer.load_state_dict(checkpoint['optim'])
        start_epoch = checkpoint['epoch']

        print("=> Start epoch {} ".format(start_epoch))

    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model, matcher)

    test_names = args.testset.strip().split(',')

    # Create data loaders
    save_path = None
    if args.gs_save:
        save_path = output_dir
    dataset, num_classes, train_loader, _, _ = get_data(args.dataset, args.data_dir, model, matcher, save_path, args)

    # Decay LR by a factor of 0.1 every step_size epochs
    lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1, last_epoch=start_epoch-1)
    best_ACC = 0.0
    if not args.evaluate:
        # Trainer
        trainer = Trainer(model, criterion, args.clip_value)
        t0 = time.time()
        # Start training
        for epoch in range(start_epoch, args.epochs):
            loss, acc = trainer.train(epoch, train_loader, optimizer)
            lr = list(map(lambda group: group['lr'], optimizer.param_groups))
            lr_scheduler.step()
            train_time = time.time() - t0
            epoch1 = epoch + 1

            print(
                '* Finished epoch %d at lr=[%g, %g, %g]. Loss: %.3f. Acc: %.2f%%. Training time: %.0f seconds.                  \n'
                % (epoch1, lr[0], lr[1], lr[2], loss, acc * 100, train_time))

            # necessary test
            if (epoch1 - start_epoch) % args.eval_interval == 0:
                for test_name in test_names:
                    if test_name not in datasets.names():
                        print('Unknown dataset: %s.' % test_name)
                        continue

                    testset, test_query_loader, test_gallery_loader = \
                        get_test_data(test_name, args.data_dir, args.height, args.width, args.workers, args.test_fea_batch)
                    test_rank1, test_mAP = evaluator.evaluate(testset, test_query_loader, test_gallery_loader)
                    acc = (test_rank1 + test_mAP) /2
                    best_ACC = max(acc, best_ACC)
                    print('%s: Testing in epoch:%d, rank1=%.1f, mAP=%.1f.\n' % ( test_name, epoch1, test_rank1 * 100, test_mAP * 100))

            save_checkpoint({
                'model': model.module.state_dict(),
                'criterion': criterion.state_dict(),
                'optim': optimizer.state_dict(),
                'epoch': epoch1,
            }, fpath=osp.join(output_dir, 'checkpoint.pth.tar'))

    json_file = osp.join(output_dir, 'results.json')
    
    if not args.evaluate:
        arg_dict = {'train_dataset': args.dataset, 'exp_dir': args.exp_dir, 'method': args.method, 'sub_method': args.sub_method}
        with open(json_file, 'a') as f:
            f.write("===================================================================================================")
            f.write('\n')
            json.dump(arg_dict, f)
            f.write('\n')
        train_dict = {'train_dataset': args.dataset, 'loss': loss, 'acc': acc, 'epochs': epoch1, 'train_time': train_time}
        with open(json_file, 'a') as f:
            json.dump(train_dict, f)
            f.write('\n')

    # Final test
    print('Evaluate the learned model:')
    t0 = time.time()

    # Evaluate

    for test_name in test_names:
        if test_name not in datasets.names():
            print('Unknown dataset: %s.' % test_name)
            continue

        t1 = time.time()
        testset, test_query_loader, test_gallery_loader = \
            get_test_data(test_name, args.data_dir, args.height, args.width, args.workers, args.test_fea_batch)


        test_rank1, test_mAP =  evaluator.evaluate(testset, test_query_loader, test_gallery_loader )
        test_time = time.time() - t1

        test_dict = {'test_dataset': test_name, 'rank1': test_rank1, 'mAP': test_mAP, 'test_time': test_time}
        print('  %s: rank1=%.1f, mAP=%.1f.\n' % (test_name, test_rank1 * 100, test_mAP * 100))
        print("best_mAcc%.1f." % (best_ACC * 100))

        with open(json_file, 'a') as f:
            json.dump(test_dict, f)
            f.write('\n')

    test_time = time.time() - t0

    if not args.evaluate:
        print('Finished training at epoch %d, loss = %.3f, acc = %.2f%%.\n'
              % (epoch1, loss, acc * 100))
        print("Total training time: %.3f sec. Average training time per epoch: %.3f sec." % (
            train_time, train_time / (epoch1 - start_epoch)))
    print("Total testing time: %.3f sec.\n" % test_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QAConv_MS")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=datasets.names(),
                        help="the training dataset")
    parser.add_argument('--combine_all', action='store_true', default=False,
                        help="combine all data for training, default: False")
    parser.add_argument('--testset', type=str, default='cuhk03_np,msmt17', help="the test datasets")
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="the batch size, default: 64")
    parser.add_argument('-j', '--workers', type=int, default=8,
                        help="the number of workers for the dataloader, default: 8")
    parser.add_argument('--height', type=int, default=384, help="height of the input image, default: 384")
    parser.add_argument('--width', type=int, default=128, help="width of the input image, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=restranmap.names(),
                        help="the backbone network, default: resnet50")
    parser.add_argument('--final_layer', type=str, default='layer3', choices=['layer2', 'layer3', 'layer4'],
                        help="the final layer, default: layer3")
    parser.add_argument('--neck', type=int, default=512,
                        help="number of channels for the final neck layer, default: 512")
    parser.add_argument('--ibn', type=str, choices={'a', 'b', 'none'}, default='b',
                        help="IBN type. Choose from 'a' or 'b'. Default: 'b'")

    # random occlusion
    parser.add_argument('--min_size', type=float, default=0, help="minimal size for the random occlusion, default: 0")
    parser.add_argument('--max_size', type=float, default=0.8, help="maximal size for the ramdom occlusion. default: 0.8")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.005,
                        help="Learning rate of the new parameters. For pretrained "
                             "parameters it is 10 times smaller than this. Default: 0.005.")
    # training configurations
    parser.add_argument('--seed', type=int, default=0) # 42 is the birth of the universe :)
    parser.add_argument('--epochs', type=int, default=27, help="the number of training epochs, default: 15")
    parser.add_argument('--step_size', type=int, default=6, help="step size for the learning rate decay, default: 10")
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help="Path for resuming training. Choices: '' (new start, default), "
                             "'ori' (original path), or a real path")
    parser.add_argument('--clip_value', type=float, default=4, help="the gradient clip value, default: 4")
    parser.add_argument('--margin', type=float, default=16, help="margin of the triplet loss, default: 16")
    parser.add_argument('--eval_interval', type=int, default=1, help="interval epoch of the test, default: 1")

    # graph sampler
    parser.add_argument('--QAConv', action='store_true', default=False, help="whether use the setup of QAConv_GS")

    parser.add_argument('--num_instance', type=int, default=4, help="the number of instance per class in a batch, default: 4")
    parser.add_argument('--gs_save', action='store_true', default=False, help="save the graph distance and top-k indices, default: False")
    parser.add_argument('--gs_verbose', action='store_true', default=False, help="verbose for the graph sampler, default: False")
    
    # test configurations
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only, default: False")
    parser.add_argument('--test_fea_batch', type=int, default=128,
                        help="Feature extraction batch size during testing. Default: 256."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_gal_batch', type=int, default=128,
                        help="QAConv gallery batch size during testing. Default: 256."
                             "Reduce this if you encounter a GPU memory overflow.")
    parser.add_argument('--test_prob_batch', type=int, default=128,
                        help="QAConv probe batch size (as kernel) during testing. Default: 256."
                             "Reduce this if you encounter a GPU memory overflow.")
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join("/data1/ckx", 'data'),
                        help="the path to the image data")
    parser.add_argument('--exp-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'Exp'),
                        help="the path to the output directory")
    parser.add_argument('--method', type=str, default='TransMatcher', help="method name for the output directory")
    parser.add_argument('--sub_method', type=str, default='res50-ibnb-layer3',
                        help="sub method name for the output directory")
    parser.add_argument('--save_score', default=False, action='store_true',
                        help="save the matching score or not, default: False")

    parser.add_argument('--nhead', type=int, default=1,
                        help="the number of heads in the multi-head-attention models (default=1)")
    parser.add_argument('--num_trans_layers', type=int, default=2,
                        help="the number of sub-encoder-layers in the encoder (default=2)")
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help="the dimension of the feedforward network model (default=2048)")
    parser.add_argument('--scale_sizes', type=str, default="1,3,5",
                        help="the multi-size of s)")
    parser.add_argument('--use_transformer', action='store_true',
                        help="whether use transformer encoder")
    parser.add_argument('--use_multi_scale', action='store_true',
                        help="whether use multi-scale template convolutions")
    parser.add_argument('--multi_scale_neck', type=int, default=512,
                        help="number of channels for the multi scale neck layer, default: 512")
    args = parser.parse_args()

    if args.QAConv:
        args.num_instance = 2
        args.clip_value = 8
        args.neck=128
    if not args.use_transformer:
        args.num_trans_layers = 0
    if not args.use_multi_scale:
        args.scale_sizes = ""
    main(args)
