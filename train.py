import argparse
import os
import random
import shutil
import time
import warnings
import sys
import math
import logging
import copy

import thop
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from autoaugment import ImageNetPolicy
from apex import amp
from model_zoo import get_model, get_model_list

import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--train-dir', type=str, default='~/data/datasets/imagenet/raw-data/train',
                        help='training pictures to use.')
    parser.add_argument('--val-dir', type=str, default='~/data/datasets/imagenet/raw-data/train',
                        help='validation pictures to use.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers(default: 4).')
    parser.add_argument('--opt_name', type=str, default='sgd',
                        help='optimizer for training, default is sgd.')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--crop-scale', type=float, default=0.08,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model for distillation training')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--classes', default=1000, type=int,
                        help='total classes to train.')
    parser.add_argument('--num-images', default=1281167, type=int,
                        help='total images to train.')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--auto-aug', action='store_true', default=False,
                        help='use autoAug for training')
    parser.add_argument('--random-erasing', action='store_true', default=False,
                        help='use random-erasing for training')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='fp16 for train')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='sync bn for train')
    opt = parser.parse_args()
    return opt


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if is_best:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state, save_dir + '/' + filename)
    else:
        return


class NewCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, sparse_label=False, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(NewCrossEntropyLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.ignore_index = ignore_index
        self.sparse_label = sparse_label

    # @weak_script_method
    def forward(self, input, target):
        if self.sparse_label:
           prob_logit = F.log_softmax(input, dim=1)
           loss = -(target * prob_logit).sum(dim=1).mean()
           return loss

        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


def adjust_learning_rate(opt, optimizer, epoch, i):
    num_iter_per_epoch = math.ceil(opt.num_images / float(opt.batch_size))
    N = num_iter_per_epoch * (opt.num_epochs - opt.warmup_epochs) - 1
    T = num_iter_per_epoch * (epoch - opt.warmup_epochs) + i

    line_slope = (opt.lr - opt.warmup_lr) / (num_iter_per_epoch * opt.warmup_epochs)

    if epoch < opt.warmup_epochs:
        lr = opt.warmup_lr + line_slope * (num_iter_per_epoch * epoch + i + 1)
    else:
        factor = (1 + math.cos(math.pi * T / N)) / 2
        lr = opt.lr * factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_data_loader(opt):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    print("--> crop_scale is %.2f"%opt.crop_scale)
    transforms_list = [transforms.RandomResizedCrop(input_size, scale=(opt.crop_scale, 1.0)),
                       transforms.RandomHorizontalFlip()]

    if opt.auto_aug:
        transforms_list.append(ImageNetPolicy())
        transforms_list.append(transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param))
    else:
        transforms_list.append(transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param))

    transforms_list.append(transforms.ToTensor())
    transforms_list.append(normalize)
    if opt.random_erasing:
        transforms_list.append(transforms.RandomErasing())

    transform_train = transforms.Compose(transforms_list)
    transform_val = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(opt.train_dir, transform_train)
    val_dataset = datasets.ImageFolder(opt.val_dir, transform_val)

    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
                   train_dataset, batch_size=opt.batch_size_per_gpu, shuffle=(train_sampler is None),
                   num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
                 val_dataset, batch_size=opt.batch_size_per_gpu, shuffle=False,
                 num_workers=opt.num_workers, pin_memory=True)
    return train_loader, val_loader, train_sampler


def train_config(opt, model):
    if (opt.label_smoothing or opt.mixup) and (not opt.evaluate):
        criterion = NewCrossEntropyLoss(sparse_label=True).cuda(opt.gpu)
    else:
        # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        criterion = NewCrossEntropyLoss().cuda(opt.gpu)

    weight, others = [],[]
    for k, v in model.named_parameters():
        if 'weight' in k:
            weight += [v]
        else:
            others += [v]

    if opt.no_wd:
        params = [{'params': weight},
                  {'params': others, 'weight_decay': 0.0}]
    else:
        params = model.parameters()

    opt_name = opt.opt_name.lower()

    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(params, opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.wd,
                                nesterov=True)

    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(params, opt.lr, weight_decay=opt.wd) 
    
    return criterion, optimizer


def load_model(opt):
    logger = opt.logger

    ### modify
    if opt.model in get_model_list() or "zennas" in opt.model or "mcunet" in opt.model:
        logger.info("=> creating model '{}'".format(opt.model))
        model = get_model(opt.model, classes=opt.classes)
    else:
        if opt.use_pretrained:
            logger.info("=> using pre-trained model '{}'".format(opt.model))
            model = models.__dict__[opt.model](pretrained=True, num_classes=opt.classes)
        else:
            logger.info("=> creating model '{}'".format(opt.model))
            model = models.__dict__[opt.model](num_classes=opt.classes)
   
    input = torch.randn(1, 3, opt.input_size, opt.input_size)
    model_copy = copy.deepcopy(model)
    opt.flops, opt.params = thop.profile(model_copy, inputs=(input, ))
    opt.flops, opt.params = thop.clever_format([opt.flops, opt.params], "%.3f")
    del model_copy
    torch.cuda.empty_cache()
    
    if opt.sync_bn:
        import apex
        logger.info("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)
 
    criterion, optimizer = train_config(opt, model)

    if opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            if opt.fp16:
                model, optimizer = amp.initialize(model, optimizer,
                                                  opt_level="O1",
                                                  loss_scale='dynamic'
                                                 )
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opt.batch_size_per_gpu = int(opt.batch_size / opt.ngpus_per_node)
            opt.num_workers = int(opt.num_workers / opt.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
        else:
            model.cuda()
            if opt.fp16:
                model, optimizer = amp.initialize(model, optimizer,
                                                  opt_level="O1",
                                                  loss_scale='dynamic'
                                                 )
            opt.batch_size_per_gpu = opt.batch_size
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if opt.model.startswith('alexnet') or opt.model.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    return model, criterion, optimizer


def mixup_transform(opt, input, target, epoch):
    lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
    if epoch >= opt.num_epochs - opt.mixup_off_epoch:
        lam = 1
   
    index = torch.randperm(input.size(0))    
    mixed_data = lam * input + (1 - lam) * input[index, :]

    if opt.label_smoothing:
        eta = 0.1
    else:
        eta = 0.0

    smoothed_labels = torch.full(size=(target.size(0), opt.classes), fill_value=eta/(opt.classes - 1)).cuda(opt.gpu, non_blocking=True)
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1-eta)
    mixed_label = lam * smoothed_labels + (1 - lam) * smoothed_labels[index, :]
    return mixed_data, mixed_label


def smooth(opt, target, eta=0.1):
    smoothed_labels = torch.full(size=(target.size(0), opt.classes), fill_value=eta/(opt.classes - 1)).cuda(opt.gpu, non_blocking=True)
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=1-eta)
    return smoothed_labels


def validate(opt, val_loader, model, criterion):
    logger = opt.logger

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    ori_flag = criterion.sparse_label
    criterion.sparse_label = False

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_interval == 0:
                logger.info(progress.print(i))

        # TODO: this should also be done with the ProgressMeter
        logger.info('**Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    criterion.sparse_label = ori_flag
    return top1.avg


def train_once(train_loader, model, criterion, optimizer, epoch, opt):
    logger = opt.logger

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
      
        adjust_learning_rate(opt, optimizer, epoch, i)

        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)

        #add mixup or label_smoothing
        if opt.mixup:
            input, hard_target = mixup_transform(opt, input, target, epoch)
        elif opt.label_smoothing:
            hard_target = smooth(opt, target)
        else:
            hard_target = target

        # compute output
        output = model(input)
        loss = criterion(output, hard_target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if opt.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_interval == 0:
            logger.info(progress.print(i))
            #logger.info('lr', optimizer.param_groups[0]['lr'])


def train(opt, train_loader, val_loader, train_sampler, model, criterion, optimizer):
    logger = opt.logger
    
    best_acc1 = 0
    for epoch in range(opt.resume_epoch, opt.num_epochs):
        if opt.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_begin_time = time.time()
        train_once(train_loader, model, criterion, optimizer, epoch, opt)
        train_end_time = time.time()

        # evaluate on validation set
        acc1 = validate(opt, val_loader, model, criterion)
        val_end_time = time.time()
        # remember best acc@1 and save checkpoint

        total_time = val_end_time-train_begin_time
        train_time = train_end_time - train_begin_time
        val_time = val_end_time - train_end_time
        logger.info('**Epoch[{}]  Total-time: {:.1f}s  Train-time: {:.1f}s  Val-time: {:.1f}s  Left-time: {:.1f}h'
                  .format(epoch, total_time, train_time, val_time, total_time*(opt.num_epochs-epoch-1)/3600))


        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not opt.multiprocessing_distributed or (opt.multiprocessing_distributed
                and opt.rank % opt.ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': opt.model,
                'state_dict': model.module.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, opt.save_dir, filename = opt.model + '_%03d'%(epoch) + '_' + '{:.4f}'.format(acc1.data) + '.pth.tar')


def main_worker(gpu, ngpus_per_node, opt):
    opt.gpu = gpu
    
    filehandler = logging.FileHandler(opt.logging_file)    
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    opt.logger = logger

    if opt.gpu is not None:
        logger.info("Use GPU: {} for training".format(opt.gpu))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            opt.rank = opt.rank * ngpus_per_node + opt.gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
    
    if (opt.rank) % opt.ngpus_per_node == 0:
        logger.setLevel(logging.INFO)
    logger.info(opt)

    model, criterion, optimizer = load_model(opt)
    logger.info(model)
    logger.info('Model FLOPS : ' + opt.flops + ', Model Params : ' + opt.params)
    logger.info('\n')

    ### modify
    if ":" in opt.model: opt.model = opt.model.split(":")[0]

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=torch.device('cpu'))
            opt.resume_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if opt.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(opt.gpu)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))
    cudnn.benchmark = True

    train_loader, val_loader, train_sampler = get_data_loader(opt)

    if opt.evaluate:
        validate(opt, val_loader, model, criterion)
        return

    train(opt, train_loader, val_loader, train_sampler, model, criterion, optimizer)


def main():
    opt = parse_args()
    ### modify
    os.makedirs(os.path.dirname(opt.logging_file), exist_ok=True)
  
    if opt.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)


if __name__ == '__main__':
    main()
