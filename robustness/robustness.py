import os
import time
import wandb
import datetime
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tqdm import tqdm

import sys

sys.path.append('..')
from read_args import get_args, print_args
from corruptions import BaseCorruption
import utils

def main_worker(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print_args(args)

    # === Dataloaders ====
    trainset, valset = utils.get_datasets(args)
    train_dataloader, val_dataloader = utils.get_dataloaders(args, trainset, valset)
    args.num_epochs = ((args.num_steps * args.gradient_acc_steps) // len(train_dataloader))

    print("Initializing Model...")
    # === Model ===
    model = utils.init_model(args)

    print('#####################################')
    print('Number of parameters ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('#####################################')

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Training Items ===
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    scheduler = utils.get_scheduler(args, optimizer)

    # === Misc ===
    run_name = utils.get_run_name(args)

    if args.gpu == 0:
        utils.init_logger(args, run_name)

    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}
    print("Checkpoint path ", args.checkpoint_path)
    utils.restart_from_checkpoint(args, 
                                    run_variables=to_restore, 
                                    model=model, 
                                    optimizer=optimizer, 
                                    scheduler=scheduler
                                    )

    start_time = time.time()

    corruptions = [
                    dict(type='CameraCrash', easy=2, mid=4, hard=5),
                    dict(type='FrameLost', easy=2, mid=4, hard=5),
                    dict(type='MotionBlur', easy=2, mid=4, hard=5),
                    dict(type='ColorQuant', easy=1, mid=2, hard=3),
                    dict(type='Brightness', easy=2, mid=4, hard=5),
                    dict(type='LowLight', easy=2, mid=3, hard=4),
                    dict(type='Fog', easy=2, mid=4, hard=5),
                    dict(type='Snow', easy=1, mid=2, hard=3)
                    ]

    img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

    # ========================== Val =================================== #

    print("Starting evaluation!")
    model.eval()
    for corruption in corruptions:
        print(f'Corruption type: {corruption["type"]}')  

        corruption_performance = []

        for severity in ['easy', 'mid', 'hard']:

            print(f'Corruption: {corruption["type"]} with severity {corruption[severity]}')
            corruptClass = BaseCorruption(
                                            severity=corruption[severity], 
                                            norm_config=img_norm_cfg, 
                                            corruption=corruption['type']
                                            )

            intersection = 0
            union = 0
            val_loader = tqdm(val_dataloader)
            for i, batch in enumerate(val_loader):

                cor = corruptClass((batch[0]*255.0))  
                batch[0] = torch.from_numpy(cor/255.0).permute(0,1,4,2,3)
                batch = [b.cuda() for b in batch]
                with torch.no_grad():
                    out = model(batch)

                intersection += out['intersection'].item()
                union += out['union'].item()

                iou = intersection / (union + 1e-4)

                # ===  Segmentation Evaluation ===
                metric_desc = f"{corruption['type']} --> mIoU: {iou * 100:.3f}"

                # === Logger ===
                val_loader.set_description(metric_desc)
                # === === ===

            logs = {}
            miou = intersection / (union + 1e-4)
            corruption_performance.append(iou)
            logs[f'{corruption}/mIoU'] = miou 
            logs[f'{corruption}/iou'] = iou

            wandb.log(logs)

        avg_corruption_performance = sum(corruption_performance) / len(corruption_performance)
        wandb.log({f'{corruption}/avg_mIoU': avg_corruption_performance})


    # === Logger ===
    print("\n=== Results ===")
    print(f"\tmIoU: {miou * 100:.3f}")
    print(f"\tIoU: {iou * 100:.3f}")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Validation time {}'.format(total_time_str))
    dist.destroy_process_group()

    return 


if __name__ == '__main__':
    args = get_args()
    main_worker(args)



