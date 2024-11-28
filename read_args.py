import sys
import math
import argparse

import torch


def set_remaining_args(args):
    args.gpus = torch.cuda.device_count()
    

    args.encoder_args = {
        'encoder_type': args.backbone,
        'resolution': args.resolution,
        'use_lora': args.use_lora,
        'lora_rank': args.lora_rank,
        'finetune': args.finetune,
    }


def print_args(args):

    print("====== Training ======")
    print(f"project name: {args.project}\n")

    print(f"backbone: {args.backbone}\n")

    print(f"model: {args.model_name}\n")

    print(f"resolution: {args.resolution}\n")

    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    print(f"effective_batch_size: {args.batch_size * args.gradient_acc_steps}")
    print(f"num_steps: {args.num_steps}")

    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Robust Camera-Based BEV Segmentation by Adapting DINOV2")

    parser.add_argument('--project', type=str, default='robust-bev')
    parser.add_argument('--model_name', type=str, default='simplebev')

    # === Model Related Parameters ===
    parser.add_argument('--backbone', type=str, default="dinov2_s", choices=["res101", "dinov2_s", "dinov2_b", "dinov2_l"])

    # finetuning
    parser.add_argument('--finetune', action="store_true")

    # adaptation setting
    parser.add_argument('--use_lora', action="store_true")
    parser.add_argument('--lora_rank', type=int, default=32)
    parser.add_argument('--use_qkv', action="store_true")

    parser.add_argument('--do_rgbcompress', action="store_true")

    # === Data Related Parameters ===
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset (e.g., /datasets/nuscenes)")
    parser.add_argument('--version', type=str, default='trainval')
    parser.add_argument('--res_scale', type=int, default=1)
    parser.add_argument('--H', type=int, default=1600)
    parser.add_argument('--W', type=int, default=900)
    parser.add_argument('--resolution',  nargs='+', type=int, default=[224, 400])
    parser.add_argument('--rand_crop_and_resize',  action="store_true")
    parser.add_argument('--rand_flip',  action="store_true")
    parser.add_argument('--cams',  nargs='+', type=str, default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])
    parser.add_argument('--ncams', type=int, default=6)
    parser.add_argument('--do_shuffle_cams',  action="store_true")
    parser.add_argument('--refcam_id', type=int, default=1)
    parser.add_argument('--bounds', nargs='+', type=int, default=[-50, 50, -5, 5, -50, 50])

    # === Log Parameters ===
    parser.add_argument('--log_freq', type=int, default=2000)
    
    # === Training Related Parameters ===
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--dropout', action="store_true")

    parser.add_argument('--batch_size', type=int, default=2
                        )
    parser.add_argument('--gradient_acc_steps', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=25000)
    parser.add_argument('--num_epochs', type=int, default=10)
   
    parser.add_argument('--seed', type=int, default=41)

    # === Misc ===
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--validate', action="store_true")
    parser.add_argument('--use_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default='./checkpoints')

    args = parser.parse_args()

    set_remaining_args(args)

    return args