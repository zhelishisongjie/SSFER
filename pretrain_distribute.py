import argparse
import os
import time
import math
import datetime

from utils.tools import seed_everything,count_parameters
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# model
from MAE_model import mae_vit_base_patch16
from MAE_model import mae_vit_small_patch16
from MAE_model import mae_vit_large_patch16
from MAE_model import mae_vit_tiny_patch16

from utils.distribute import get_rank,get_world_size,init_distributed_mode,is_main_process




def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # train
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # optimizer
    parser.add_argument('--warmup_epochs', type=int, default=50, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--lr', type=float, default=3.4e-4, metavar='LR',help='learning rate (absolute lr)')
    parser.add_argument('--weight_decay', type=float, default=0.05,help='weight decay (default: 0.05)')

    # model
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--mask_ratio', default=0.75, type=float,help='Masking ratio (percentage of removed patches).')

    # dataset
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--resume', default='./output_dir/125w_base_warmup50_batch256_280.pth', help='resume from checkpoint') # resume = 'mae_tiny_RAFDB_2.pth'     # resume from checkpoint
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--output_dir', default='./output_dir',help='path where to save, empty for no saving')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def main(args):
    init_distributed_mode(args)

    device = torch.device(args.device)

    # seed
    seed_everything(args.seed)

    # dir
    # traindir = './RAF-DB/train'
    traindir = './pretrain_dataset/train_all'


    model = mae_vit_base_patch16()
    model.to(args.device)

    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / args.epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=False)



    transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
      				  transforms.Normalize(mean=[0.588,0.459,0.402], std=[0.228, 0.202, 0.190])
    ])
      					
      					
      				  

    train_dataset = datasets.ImageFolder(traindir , transform = transform)
    print(len(train_dataset))

    num_tasks = get_world_size()
    global_rank = get_rank()
    # sampler_train = torch.utils.data.RandomSampler(train_dataset)  # 和shuffle类似
    sampler_train = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    train_loader = DataLoader(train_dataset, sampler = sampler_train, batch_size = args.batch_size , num_workers = args.num_workers, pin_memory = True,   drop_last = True,)




    def train(train_loader, epochs, resume):
        train_loss_list = []
        step_count = 0

        if resume:
            checkpoint = torch.load(resume, map_location='cpu')
            start_epoch = checkpoint['epoch']
            model.module.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"============== Resume training for {epochs} epochs from {start_epoch} ==============")
        else:
            start_epoch = 0
            print(f"============== Start training for {epochs} epochs ==============")

        optimizer.zero_grad()
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch: {start_epoch} , start lr: {lr}")
        for epoch in range(start_epoch, epochs):

            model.train()
            losses = []
            for img, labels in train_loader:
                step_count += 1
                img = img.to(device)

                loss, _, _ = model(img, mask_ratio=args.mask_ratio)

                loss.backward()  # 反向传播：计算参数梯度值
                if step_count % args.accum_iter == 0:
                    optimizer.step()  # 梯度下降：参数更新
                    optimizer.zero_grad()  # 梯度归0

                losses.append(loss.item())

                torch.cuda.synchronize()  # cpu gpu 同步

            lr_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            avg_loss = sum(losses) / len(losses)

            print(f"========================== epoch:{epoch + 1}/{epochs} ==========================")
            print(f"loss: {avg_loss}   lr: {lr}")
            train_loss_list.append(avg_loss)

            if (epoch + 1) % 20 == 0 and is_main_process():
                state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                         'lr_scheduler_state_dict': lr_scheduler.state_dict()}
                torch.save(state, f'./output_dir/125w_base_warmup50_batch256_{epoch + 1}.pth')
        return train_loss_list



    '''--------------------------------train----------------------------------'''
    start_time = time.time()
    train_loss_list = train(train_loader , args.epochs , args.resume)
    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
    '''-----------------------------------------------------------------------'''


    '''---------------------------------plot-----------------------------------'''
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Value')

    plt.savefig(f'vit_base_loss_{args.epochs}.png', dpi=800)





if __name__ == '__main__':
    # os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:21'
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)