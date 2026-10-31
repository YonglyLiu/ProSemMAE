import os
import time
import math
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.med_datasets import LargeMedicalDataSets
# from dataloader.med_datasets import BTCVDataset
from prosemmae import build_vit_large_prosemmae_p12_3d, build_vit_base_prosemmae_p16_3d, build_vit_large_prosemmae_p16_3d, build_vit_large_mae_p16_3d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import SequentialSampler

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

huber_loss = nn.HuberLoss(delta=1.0)


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


class ProSemMAETrainer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_name = 'ProSemMAE'
    
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        
        if self.world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
            self.distributed = True
        else:
            self.distributed = False
        
        
        if self.local_rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=args.log_dir)
        else:
            self.writer = None
            
        


        # scaler for AMP (each process maintains its own scaler)
        self.scaler = torch.cuda.amp.GradScaler()

        # set some attributes and build components
        self.init_lr()
        self.build_model()
        self.build_optimizer()
        self.build_dataloader()

        # iterations per epoch (use dataloader length)
        self.iters_per_epoch = len(self.dataloader) if self.dataloader is not None else 0

        if self.local_rank == 0:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir, exist_ok=True)
            print("mask ratio: {}".format(args.mask_ratio))
            print(f"Running on: {'Distributed (DDP)' if self.distributed else 'Single GPU/CPU'}")
            print("Mask ratio: {}".format(args.mask_ratio))
            print(f"Local Batch Size: {args.batch_size}")

    def init_lr(self):
        args = self.args
        # infer learning rate before changing batch size
        self.lr = args.base_lr

    def build_model(self):
        args = self.args

        if args.model == "prosemmae_vit_large_12p":
            model = build_vit_large_prosemmae_p12_3d(args=args).to(self.device)
        elif args.model == "prosemmae_vit_base_16p":
            model = build_vit_base_prosemmae_p16_3d(args=args).to(self.device)
        elif args.model == "prosemmae_vit_large_16p":
            model = build_vit_large_prosemmae_p16_3d(args=args).to(self.device)
        elif args.model == "mae_vit_large_16p":
            model = build_vit_large_mae_p16_3d(args=args).to(self.device)
        else:
            model = None
        # self.model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        model.to(self.device)
        if self.distributed:
            # When using DDP, use device_ids=[local_rank] and find_unused_parameters if necessary
           
            
            model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)



        
        self.model = model
        

    def build_optimizer(self):
        args = self.args

        optim_params = self.get_param_groups(
            nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'decoder_pos_embed', 'gamma'})
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params,
                                           lr=self.lr,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    def build_dataloader(self):
        args = self.args
        train_dataset = LargeMedicalDataSets(args=args)
        print("train_dataset longggg", len(train_dataset))
        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True)
        else:
            train_sampler = SequentialSampler(train_dataset)
        #train_sampler = SequentialSampler(train_dataset)
        self.dataloader = DataLoader(train_dataset, sampler=train_sampler,
                             batch_size=args.batch_size,
                             num_workers=args.num_worker,
                             pin_memory=True)
    
    
    
    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = 0
        # for epoch in range(args.remain, args.epochs):
        for epoch in range(0, args.epochs):
            if self.distributed and isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(epoch)
            if (epoch + 1) % 50 == 0:
                if not os.path.exists(args.save_ckpt_dir):
                    # os.mkdir(args.ckpt_dir)
                    os.makedirs(args.save_ckpt_dir, exist_ok=True)
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                }, is_best=False, filename=f'{args.save_ckpt_dir}/{args.model}_ckpt_{epoch:04d}.pth')
                print("=> finish saving checkpoint")

            # train for one epoch
            # self.dataloader.sampler.set_epoch(epoch)
            # self.dataloader.sampler(epoch)
            niters = self.epoch_train(epoch, niters)
        if self.distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()   


    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        print(f"Length of train_loader: {len(train_loader)}")
        print(train_loader)

        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        loss1 = AverageMeter()
        load_time = 0
        forward_time = 0
        bp_time = 0
        cache2gpu_time = 0
        # switch to train mode
        model.train()
        load_start_time = time.time()
        start_time = time.time()

        for i, batch_data in enumerate(train_loader):
            load_time += time.time() - load_start_time

            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            gpu_time = time.time()

            batch_data = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in crop.items()} for
                          crop in batch_data]

            image = torch.stack([crop["image"] for crop in batch_data], dim=0).view(-1, 1, 96, 96, 96).to(self.device)
            cache2gpu_time += time.time() - gpu_time

            forward_start_time = time.time()
            with torch.cuda.amp.autocast(False):
                loss = model(image, return_image=False)
                #loss, img, recon, masked_x = model(image, return_image=True)

                loss1.update(loss.item(), image.size(0))
            forward_time += time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time += time.time() - bp_start_time
            niters += 1
            
            if self.writer is not None:
                self.writer.add_scalar("train/loss", loss.item(), niters)

            
            load_start_time = time.time()
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.epochs, i, len(train_loader)),
                "loss: {:.4f}".format(loss),
                "time {:.2f}s".format(time.time() - start_time))
                

        end_time = time.time()
        duration = end_time - start_time
        # Log to the screen

        print(
            "Epoch: {}/{} took {:.2f} seconds | TotalIter {}/{} | Init Lr: {:.6f} | Lr: {:.6f} | Load Time: {:.2f} s | GPU Time: {:.2f} s | Forward Time: {:.2f} s | Backward Time: {:.2f} s | Loss: {:.4f}"
            .format(epoch, args.epochs, duration, niters, args.epochs * self.iters_per_epoch, self.lr,
                    optimizer.param_groups[0]['lr'],
                    load_time,
                    cache2gpu_time,
                    forward_time,
                    bp_time,
                    loss1.avg))
        return niters

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth'):
        if self.distributed and self.local_rank != 0:
            return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth')

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (
                        1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

    def get_param_groups(self, nowd_keys=()):
        para_groups, para_groups_dbg = {}, {}

        for name, para in self.model.named_parameters():
            if not para.requires_grad:
                continue  # frozen weights
            if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
                wd_scale, group_name = 0., 'no_decay'
            else:
                wd_scale, group_name = 1., 'decay'

            if group_name not in para_groups:
                para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
                para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': 1.}
            para_groups[group_name]['params'].append(para)
            para_groups_dbg[group_name]['params'].append(name)

        return list(para_groups.values())
