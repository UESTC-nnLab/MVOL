import argparse
import os
import random
import shutil
import time
import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
from tensorboard_logger import configure, log_value

import models.densenet as dn
import models.ensemble_net as en
import models.wideresnet as wn
from utils import CIFAR10_Subset, KD_CIFAR10, KD_CIFAR100, KD_CIFAR10_Subset, RandomTinyImages300k, Mixed_CIFAR100

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpu', default='0', type=str, help='which gpu to use')
parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
parser.add_argument('--model-arch', default='densenet', type=str, help='model architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--save-epoch', default=10, type=int, help='save the model every save_epoch')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, help='mini-batch size for in-distribution data')
parser.add_argument('--ood-batch-size', default=64, type=int, help='mini-batch size for out-of-distribution data')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float, 
                    help='weight decay (default: 0.0001 for densenet, 0.0005 for wideresnet)')
parser.add_argument('--print-freq', '-p', default=100, type=int, help='print frequency')
parser.add_argument('--layers', default=100, type=int, help='total number of layers of densenet')
parser.add_argument('--depth', default=40, type=int, help='depth of resnet')
parser.add_argument('--width', default=2, type=int, help='width of resnet')
parser.add_argument('--growth', default=12, type=int, help='number of new channels per layer (default: 12) for densenet')
parser.add_argument('--reduce', default=0.5, type=float, 
                    help='compression rate in transition stage (default: 0.5) for densenet')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability (default: 0.0)')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--exp-cat-name', required=True, type=str, help='the name of experience categories, e.g., oe')
parser.add_argument('--tensorboard', default=True, help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--seed', required=True, type=int, help='random initialization of network')

parser.add_argument('--name', required=True, type=str, help='name of experiment')
parser.add_argument('--teacher-names', required=True, type=str, help='names of teacher models')
parser.add_argument('--teacher-epochs', default=100, type=int, help='weights of teacher models')
parser.add_argument('--alpha', default=0.5, type=float, help='hyper-parameter alpha of knowledge distillation')
parser.add_argument('--temperature', default=2.0, type=float, help='hyper-parameter temperature of knowledge distillation')
parser.add_argument('--inference-batch-size', default=100, type=int, help='mini-batch size for inference of teachers')

parser.add_argument('--noise', default=0, required=False, type=float, help='noisy degree in wild settings')
parser.add_argument('--auxiliary-dataset', default='RandomImages300k', required=True, 
                    choices=['mixed_cifar100', 'RandomImages300k'], type=str, help='which auxiliary dataset to use')
parser.add_argument('--beta', default=0.5, type=float, help='loss weight in OE')                
parser.add_argument('--factor', default=1., type=float, help='used in mvol')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)
directory = "checkpoints/{exp_cat_name}/{in_dataset}/{auxiliary_dataset}/{model_arch}/{name}/".\
    format(exp_cat_name=args.exp_cat_name, in_dataset=args.in_dataset, model_arch= args.model_arch, auxiliary_dataset = args.auxiliary_dataset, name=args.name)

if not os.path.exists(directory):
    os.makedirs(directory)
save_state_file = os.path.join(directory, 'args.txt')
fw = open(save_state_file, 'w')
print(state, file=fw)
fw.close()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def set_seeds(seed:int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def loss_fn_kd(outputs, labels, logits, alpha = 0.5, temperature = 2):
    """
    knowledge distillation criterion 
    """
    T = temperature
    criterion = nn.CrossEntropyLoss().cuda()

    predictions = torch.exp(logits / T) / torch.sum(torch.exp(logits / T), dim=-1, keepdim=True)
    mean_predictions = torch.mean(predictions, dim=1)

    kd_loss = criterion(outputs/T, mean_predictions)
    ce_loss = criterion(outputs, labels)

    loss = kd_loss * (alpha * T * T) + ce_loss * (1. - alpha)

    return loss, ce_loss

def main():
    if args.tensorboard: configure("runs/%s/%s/%s/%s/%s"%(args.exp_cat_name, args.in_dataset, args.auxiliary_dataset, args.model_arch, args.name))
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_teacher_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),    
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        teacher_id_train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_teacher_train),
            batch_size=args.inference_batch_size, shuffle=False, **kwargs)
        teacher_id_val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./datasets/cifar10', train=False, transform=transform_teacher_train),
            batch_size=args.inference_batch_size, shuffle=False, **kwargs)      

        lr_schedule=[50, 75, 90]
        num_classes = 10
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        teacher_id_train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                             transform=transform_teacher_train),
            batch_size=args.inference_batch_size, shuffle=False, **kwargs)
        teacher_id_val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./datasets/cifar100', train=False, transform=transform_teacher_train),
            batch_size=args.inference_batch_size, shuffle=False, **kwargs)

        lr_schedule=[50, 75, 90]
        num_classes = 100
    elif args.in_dataset =="CIFAR-10_subset":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        teacher_id_train_loader = torch.utils.data.DataLoader(
            CIFAR10_Subset('./datasets/cifar10', train=True, download=True,
                             transform=transform_train),
            batch_size=args.inference_batch_size, shuffle=False, **kwargs)
        teacher_id_val_loader = torch.utils.data.DataLoader(
            CIFAR10_Subset('./datasets/cifar10', train=False, transform=transform_teacher_train),
            batch_size=args.inference_batch_size, shuffle=False, **kwargs)
        
        lr_schedule=[50, 75, 90]
        num_classes = 10

    set_seeds(args.seed)

    teacher_model = en.EnsembleNet(teacher_epochs = args.teacher_epochs, model_arch = args.model_arch, 
                       teacher_names= args.teacher_names, num_classes = num_classes, 
                       normalizer = normalizer, in_dataset=args.in_dataset)
    teacher_model.eval()
    teacher_model.cuda()

    # create student model
    if args.model_arch == 'densenet':
        model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
    elif args.model_arch == 'wideresnet':
        model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, dropRate=args.droprate, normalizer=normalizer)
    else:
        assert False, 'Not supported model arch: {}'.format(args.model_arch)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    model = model.cuda()

    criterion = loss_fn_kd
    oe_criterion = MVOLLoss(factor=args.factor).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    
    ##### teacher model predictions
    teacher_id_train_outputs = generate_KD_label(teacher_loader = teacher_id_train_loader, teacher_model = teacher_model)
    teacher_id_val_outputs = generate_KD_label(teacher_loader=teacher_id_val_loader, teacher_model = teacher_model)

    if args.in_dataset == "CIFAR-10":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        student_id_train_loader = torch.utils.data.DataLoader(
            KD_CIFAR10('./datasets/cifar10', train=True, download=True,
                             transform=transform_train,teacher_outputs=teacher_id_train_outputs),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        student_id_val_loader = torch.utils.data.DataLoader(
            KD_CIFAR10('./datasets/cifar10', train=False, transform=transform_test, teacher_outputs=teacher_id_val_outputs),
            batch_size=args.batch_size, shuffle=False, **kwargs)      
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        student_id_train_loader = torch.utils.data.DataLoader(
            KD_CIFAR100('./datasets/cifar100', train=True, download=True,
                             transform=transform_train, teacher_outputs=teacher_id_train_outputs),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        student_id_val_loader = torch.utils.data.DataLoader(
            KD_CIFAR100('./datasets/cifar100', train=False, transform=transform_test, teacher_outputs=teacher_id_val_outputs),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset =="CIFAR-10_subset":
        # Data loading code
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        student_id_train_loader = torch.utils.data.DataLoader(
            KD_CIFAR10_Subset('./datasets/cifar10', train=True, download=True,
                             transform=transform_train, teacher_outputs=teacher_id_train_outputs),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        student_id_val_loader = torch.utils.data.DataLoader(
            KD_CIFAR10_Subset('./datasets/cifar10', train=False, transform=transform_test, teacher_outputs=teacher_id_val_outputs),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        assert False, 'Not supported dataset: {}'.format(args.in_dataset)

    if args.auxiliary_dataset == 'RandomImages300k':
        student_oe_train_loader = torch.utils.data.DataLoader(RandomTinyImages300k(transform=transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor()])),
            batch_size=args.ood_batch_size, shuffle= False, **kwargs)
    elif args.auxiliary_dataset == 'mixed_cifar100':
        student_oe_train_loader = torch.utils.data.DataLoader(
        Mixed_CIFAR100(cifar10_root ='../datasets/CIFAR-10', cifar100_root = '../datasets/CIFAR-100',transform=transforms.Compose(
            [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor()]), download=True, mixed_ratio=args.noise, 
            ), batch_size=args.ood_batch_size, shuffle=False,**kwargs
        )
    else:
        assert False, 'Not supported dataset: {}'.format(args.in_dataset)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_schedule)

        # train for one epoch
        train_oe(student_id_train_loader, student_oe_train_loader, model, criterion, oe_criterion, optimizer, epoch)

        # evaluate on validation set
        validate(student_id_val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        if (epoch + 1) % args.save_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
            }, epoch + 1)

def generate_KD_label(teacher_loader, teacher_model):
    teacher_output_list = []
    with tqdm.tqdm(total=len(teacher_loader), ) as t:
        for i, (input, target) in enumerate(teacher_loader):
            input = input.cuda()
            with torch.no_grad():
                output_teacher_batch = teacher_model.get_ensemble_outputs(input)
            teacher_output_list.append(output_teacher_batch.detach().cpu())
            t.update(1)
            t.desc = 'teacher model processing'
    return torch.cat(teacher_output_list, dim = 0)

def train_oe(student_id_train_loader, student_oe_train_loader, model, criterion, oe_criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    in_losses = AverageMeter()
    out_losses = AverageMeter()
    ce_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (in_set, out_set) in enumerate(zip(student_id_train_loader, student_oe_train_loader)):

        in_len = len(in_set[0])
        out_len = len(out_set[0])
        target = in_set[1]
        target = target.cuda()
        teatcher_outputs_batch = in_set[2]
        teatcher_outputs_batch = teatcher_outputs_batch.cuda()

        input = torch.cat((in_set[0], out_set[0]), 0).cuda()

        #input = input.detach().clone()
        output = model(input)

        in_output = output[:in_len]
        # in_loss = criterion(in_output, target)
        kd_in_loss, ce_loss = criterion(in_output, target, teatcher_outputs_batch, alpha=args.alpha, temperature = args.temperature)

        out_output = output[in_len:]
        out_loss = oe_criterion(out_output)

        # measure accuracy and record loss
        prec1 = accuracy(in_output.data, target, topk=(1,))[0]
        in_losses.update(kd_in_loss.data, in_len)
        out_losses.update(out_loss.data, out_len)
        ce_losses.update(ce_loss.data, in_len)
        top1.update(prec1, in_len)

        # compute gradient and do SGD step
        loss = kd_in_loss + args.beta * out_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'In Loss {in_loss.val:.4f} ({in_loss.avg:.4f})\t'
                  'Out Loss {out_loss.val:.4f} ({out_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(student_id_train_loader), batch_time=batch_time,
                      in_loss=in_losses, out_loss=out_losses, top1=top1))

    # log to TensorBoardss
    if args.tensorboard:
        log_value('train_acc', top1.avg, epoch)
        log_value('out_loss', out_losses.avg, epoch)
        log_value('in_loss', in_losses.avg, epoch)
        log_value('train_ce_loss', ce_losses.avg, epoch)

def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, output_teacher_batch) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output_teacher_batch = output_teacher_batch.cuda()    
        with torch.no_grad():
            output = model(input)

        kd_loss, ce_loss = criterion(output, target, output_teacher_batch, alpha=args.alpha, temperature = args.temperature)        

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(kd_loss.data, input.size(0))
        ce_losses.update(ce_loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
        log_value('val_ce_loss', ce_losses.avg, epoch)
    return losses.avg, top1.avg

class MVOLLoss(nn.Module):
    def __init__(self, factor):
        super(MVOLLoss, self).__init__()
        self.CE = torch.nn.CrossEntropyLoss()
        self.factor = factor
    def forward(self, x):
        prob = F.softmax(x, dim = 1).detach().double()
        prob = torch.where(prob > self.factor * 1/x.shape[1], self.factor * 1/x.shape[1], prob)
        loss = self.CE(x, prob)
        return loss

def save_checkpoint(state, epoch):
    """Saves checkpoint to disk"""
    directory = "checkpoints/{exp_cat_name}/{in_dataset}/{auxiliary_dataset}/{model_arch}/{name}/"\
                 .format(exp_cat_name = args.exp_cat_name, in_dataset=args.in_dataset, model_arch= args.model_arch, auxiliary_dataset = args.auxiliary_dataset, name=args.name)    
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + 'checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """Sets the learning rate to the initial LR decayed by 10 after 40 and 80 epochs"""
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()