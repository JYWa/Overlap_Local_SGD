import os
import numpy as np
import time
import argparse
import sys

from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.multiprocessing import Process
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models

from distoptim import LocalSGD, OverlapLocalSGD
import util_v4 as util
from comm_helpers import SyncAllreduce

parser = argparse.ArgumentParser(description='CIFAR-10 baseline')
parser.add_argument('--name','-n', 
                    default="default", 
                    type=str, 
                    help='experiment name, used for saving results')
parser.add_argument('--backend',
                    default="nccl",
                    type=str,
                    help='experiment name, used for saving results')
parser.add_argument('--dataset', 
                    default="cifar10", 
                    type=str, 
                    help='dataset name')
parser.add_argument('--model', 
                    default="res", 
                    type=str, 
                    help='neural network model')
parser.add_argument('--alpha', 
                    default=0.2, 
                    type=float, 
                    help='alpha')
parser.add_argument('--gmf', 
                    default=0, 
                    type=float, 
                    help='global momentum factor')
parser.add_argument('--lr', 
                    default=0.1, 
                    type=float, 
                    help='learning rate')
parser.add_argument('--bs', 
                    default=512, 
                    type=int, 
                    help='batch size on each worker')
parser.add_argument('--epoch', 
                    default=200, 
                    type=int, 
                    help='total epoch')
parser.add_argument('--cp', 
                    default=98, 
                    type=int, 
                    help='communication period / work per clock')
parser.add_argument('--print_freq', 
                    default=100, 
                    type=int, 
                    help='print info frequency')
parser.add_argument('--rank', 
                    default=0, 
                    type=int, 
                    help='the rank of worker')
parser.add_argument('--size', 
                    default=8, 
                    type=int, 
                    help='number of workers')
parser.add_argument('--seed', 
                    default=1, 
                    type=int, 
                    help='random seed')
parser.add_argument('--save', '-s', 
                    action='store_true', 
                    help='whether save the training results')
parser.add_argument('--all_reduce',
                    action='store_true', 
                    help='whether use AR-SGD')
parser.add_argument('--schedule', nargs='+', default=None,
                    type=float, help='learning rate schedule')
parser.add_argument('--warmup', default='False', type=str,
                    help='whether to warmup learning rate for first 5 epochs')
parser.add_argument('--p', '-p', 
                    action='store_true', 
                    help='whether the dataset is partitioned or not')

parser.add_argument('--NIID',
                    action='store_true',
                    help='whether the dataset is partitioned or not')

args = parser.parse_args()
args.lr_schedule = {}
if args.schedule is None:
    args.schedule = [30, 0.1, 60, 0.1, 80, 0.1]
i, epoch = 0, None
for v in args.schedule:
    if i == 0:
        epoch = v
    elif i == 1:
        args.lr_schedule[epoch] = v
    i = (i + 1) % 2
del args.schedule
print(args)

def run(rank, size):
    # initiate experiments folder
    save_path = '/users/jianyuw1/SGD_non_iid/results/'
    folder_name = save_path+args.name
    if rank == 0 and os.path.isdir(folder_name)==False and args.save:
        os.mkdir(folder_name)
    dist.barrier()
    # initiate log files
    tag = '{}/lr{:.3f}_bs{:d}_cp{:d}_a{:.2f}_b{:.2f}_e{}_r{}_n{}.csv'
    saveFileName = tag.format(folder_name, args.lr, args.bs, args.cp, 
                              args.alpha, args.gmf, args.seed, rank, size)
    args.out_fname = saveFileName
    with open(args.out_fname, 'w+') as f:
        print(
            'BEGIN-TRAINING\n'
            'World-Size,{ws}\n'
            'Batch-Size,{bs}\n'
            'Epoch,itr,BT(s),avg:BT(s),std:BT(s),'
            'CT(s),avg:CT(s),std:CT(s),'
            'Loss,avg:Loss,Prec@1,avg:Prec@1,val'.format(
                ws=args.size,
                bs=args.bs),
            file=f)


    # seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # load datasets
    train_loader, test_loader = util.partition_dataset(rank, size, args)

    # define neural nets model, criterion, and optimizer
    model = util.select_model(10, args).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = LocalSGD(model.parameters(),
                      lr=args.lr,
                      gmf=args.gmf,
                      tau=args.cp
                      size=size,
                      momentum=0.9,
                      nesterov = True,
                      weight_decay=1e-4)

    # optimizer = OverlapLocalSGD(model.parameters(),
    #                       lr=args.lr,
    #                       alpha=args.alpha,
    #                       gmf=args.gmf,
    #                       tau = args.cp,
    #                       size=size,
    #                       momentum=0.9,
    #                       nesterov = True,
    #                       weight_decay=1e-4)

    batch_meter = util.Meter(ptag='Time')
    comm_meter = util.Meter(ptag='Time')

    best_test_accuracy = 0
    req = None
    for epoch in range(args.epoch):
        train(model, criterion, optimizer, batch_meter, comm_meter,
              train_loader, epoch)
        test_acc = evaluate(model, test_loader)
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc

        with open(args.out_fname, '+a') as f:
            print('{ep},{itr},{bt:.4f},{filler},{filler},'
                  '{ct:.4f},{filler},{filler},'
                  '{filler},{filler},'
                  '{filler},{filler},'
                  '{val:.4f}'
                  .format(ep=epoch, itr=-1,
                          bt=batch_meter.sum,
                          ct=comm_meter.sum,
                          filler=-1, val=test_acc), 
                  file=f)


def evaluate(model, test_loader):
    model.eval()
    top1 = util.AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)
            outputs = model(data)
            acc1 = util.comp_accuracy(outputs, target)
            top1.update(acc1[0].item(), data.size(0))

    return top1.avg

def train(model, criterion, optimizer, batch_meter, comm_meter,
          loader, epoch):

    model.train()

    losses = util.Meter(ptag='Loss')
    top1 = util.Meter(ptag='Prec@1')
    weights = [1/args.size for i in range(args.size)]

    iter_time = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        # data loading
        data = data.cuda(non_blocking = True)
        target = target.cuda(non_blocking = True)

        # forward pass
        output = model(data)
        loss = criterion(output, target)

        # backward pass
        loss.backward()
        update_learning_rate(optimizer, epoch, itr=batch_idx,
                                 itr_per_epoch=len(loader))
        # gradient step
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        comm_start = time.time()
        
        # Communication step: average local models
        optimizer.average()

        if not (epoch == 0 and batch_idx == 0):
            torch.cuda.synchronize()
            comm_meter.update(time.time() - comm_start)
            batch_meter.update(time.time() - iter_time)

        # write log files
        train_acc = util.comp_accuracy(output, target)
        losses.update(loss.item(), data.size(0))
        top1.update(train_acc[0].item(), data.size(0))

        if batch_idx % args.print_freq == 0 and args.save:
            print('epoch {} itr {}, '
                  'rank {}, loss value {:.4f}, train accuracy {:.3f}'.format(
                    epoch, batch_idx, rank, losses.avg, top1.avg))

            with open(args.out_fname, '+a') as f:
                print('{ep},{itr},{bt},{ct},'
                      '{loss.val:.4f},{loss.avg:.4f},'
                      '{top1.val:.3f},{top1.avg:.3f},-1'
                      .format(ep=epoch, itr=batch_idx,
                              bt=batch_meter, ct=comm_meter,
                              loss=losses, top1=top1), file=f)

        torch.cuda.synchronize()
        iter_time = time.time()

    with open(args.out_fname, '+a') as f:
        print('{ep},{itr},{bt},{ct},'
              '{loss.val:.4f},{loss.avg:.4f},'
              '{top1.val:.3f},{top1.avg:.3f},-1'
              .format(ep=epoch, itr=batch_idx,
                      bt=batch_meter, ct=comm_meter,
                      loss=losses, top1=top1), file=f)


def update_learning_rate(optimizer, epoch, itr=None, itr_per_epoch=None,
                         scale=1):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially (epochs 30, 60, 80)
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    target_lr = args.lr * args.bs * scale * args.size / 128

    lr = None
    if args.warmup and epoch < 5:  # warmup to scaled lr
        if target_lr <= args.lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - args.lr) * (count / (5 * itr_per_epoch))
            lr = args.lr + incr
    else:
        lr = target_lr
        for e in args.lr_schedule:
            if epoch >= e:
                lr *= args.lr_schedule[e]

    if lr is not None:
        # print('Updating learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def init_processes(rank, size, fn):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend=args.backend, 
                            init_method='tcp://h0:22000', 
                            rank=rank, 
                            world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    rank = args.rank
    size = args.size
    print(rank)
    init_processes(rank, size, run)


