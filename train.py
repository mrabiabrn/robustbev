import os
import time
import wandb
import datetime

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tqdm import tqdm


import utils
from read_args import get_args, print_args



def train_epoch(args, model, optimizer, scheduler, train_dataloader, val_dataloader, total_iter):
    
    total_loss = 0 

    loader = tqdm(train_dataloader, disable=(args.gpu != 0))

    for i, batch in enumerate(loader):
        model.train()

        logs = {} 

        # === Update time ====
        if (i) % args.gradient_acc_steps == 0:

            batch = [b.cuda() for b in batch]

            out = model(batch)

            loss = out['total_loss']

            loss /= args.gradient_acc_steps
            total_loss += loss.item()
            loss.backward()
    
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            total_iter += 1

            if args.gpu == 0:
                lr = optimizer.state_dict()["param_groups"][0]["lr"]
                mean_loss = total_loss / (i + 1)
                loader.set_description(f"lr: {lr:.6f} | loss: {mean_loss:.5f} | total_iter: {total_iter}")

                logs = {
                        'total iter': total_iter,
                        'loss': mean_loss, 
                        'lr':lr
                     }
                
                loss_details = out['loss_details']
                for k, v in loss_details.items():
                    logs[f"loss/{k}"] = v

                if total_iter % args.log_freq == 0 or total_iter == (len(train_dataloader) * args.num_epochs // args.gradient_acc_steps) - 1:

                    train_subset_loader = utils.get_random_subset_dataloader(train_dataloader.dataset, n=1000)
                    train_logs, _ = eval(model, train_subset_loader)
                    for k, v in train_logs.items():
                        logs[f"train/{k}"] = v

                    val_logs, val_loss = eval(model, val_dataloader)
                    for k, v in val_logs.items():
                        logs[f"val/{k}"] = v
                        logs["val/loss"] = val_loss

        # === Just Calculate ====
        else:
            with model.no_sync():
                batch = [b.cuda() for b in batch]
                out = model(batch)

                loss = out['total_loss']

                loss /= args.gradient_acc_steps
                total_loss += loss.item()
                loss.backward()

             
        # === Log ===
        if args.gpu == 0:
            if logs:
                wandb.log(logs)


    statistics = {
        'loss': total_loss / (i + 1),
    }
   
    return statistics, total_iter




@torch.no_grad()
def eval(model, val_dataloader):

    model.eval()

    val_loader = tqdm(val_dataloader)
    
    total_loss = 0

    intersection = 0
    union = 0

    for i, batch in enumerate(val_loader):

        batch = [b.cuda() for b in batch]

        out = model(batch)

        intersection += out['intersection'].item()
        union += out['union'].item()
        miou_cur = intersection / (1e-6 + union)

        total_loss += out['total_loss']
    
        # ===  Segmentation Evaluation ===
        metric_desc = f"mIoU: {miou_cur * 100:.3f}"

        # === Logger ===
        val_loader.set_description(metric_desc)
        # === === ===

    # === Evaluation Results ====
    miou = intersection / (1e-6 + union)
    total_loss =  total_loss / (i+1)

    # === Logger ===
    print("\n=== Results ===")
    print(f"\tmIoU: {miou * 100:.3f}")

    logs = {
        'mIoU': miou,
    }

    return logs, total_loss



def main_worker(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    print_args(args)
   
    # === Model ===
    model = utils.init_model(args)

    utils.print_model_summary(args, model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Dataloaders ====
    trainset, valset = utils.get_datasets(args)
    train_dataloader, val_dataloader = utils.get_dataloaders(args, trainset, valset)

    # === Epochs ===
    args.num_epochs = ((args.num_steps * args.gradient_acc_steps) // len(train_dataloader))

    print('#########################################')
    print('It will take {} epochs to complete'.format(args.num_epochs))
    print('#########################################')

    # === Training Items ===
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    scheduler = utils.get_scheduler(args, optimizer)

    print('#########################################')
    print(f"Optimizer and schedulers ready.")
    print('#########################################')

    # === Misc ===
    run_name = utils.get_run_name(args)

    if args.gpu == 0:
        utils.init_logger(args, run_name)

    # === Load from Checkpoint ===
    to_restore = {"epoch": 0}
    if args.use_checkpoint:
        utils.restart_from_checkpoint(args, 
                                      run_variables=to_restore, 
                                      model=model, 
                                      optimizer=optimizer, 
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]


    start_time = time.time()

    dist.barrier()

    # ========================== Val =================================== #
    if args.validate:

        print("Starting evaluation!")
        if args.gpu == 0:
            eval(model, val_dataloader)

        dist.barrier()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
        dist.destroy_process_group()
        return

    # ========================== Train =================================== #

    if args.gpu == 0:
        if not os.path.exists(os.path.join(args.model_save_path, run_name)):
            os.makedirs(os.path.join(args.model_save_path, run_name))

    print("Starting training!")

    total_iter = 0

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        statistics, total_iter = train_epoch(args, model, optimizer, scheduler, train_dataloader, val_dataloader, total_iter)

        if args.gpu == 0:

            if epoch % args.save_epoch == 0 or epoch == args.num_epochs - 1:

                # === Save Checkpoint ===
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "args": args,
                }

                utils.save_on_master(save_dict, os.path.join(args.model_save_path, run_name, f"{epoch}.pt"))
                print(f"Model saved at epoch {epoch}")

        dist.barrier()

        print("===== ===== ===== ===== =====")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args()
    main_worker(args)



