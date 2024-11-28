#!/bin/bash

echo "Training..."

torchrun --master_port=1303 --nproc_per_node=1 "robustness.py" \
                                                                                     --dataset_path "/datasets/nuscenes/" \
                                                                                     --batch_size 8 \
                                                                                     --backbone "res101" \
                                                                                     --resolution 224 400 \
                                                                                     --ncams 6 \
                                                                                     --do_rgbcompress \
                                                                                     --gradient_acc_steps 5 \
                                                                                     --rand_flip \
                                                                                     --rand_crop_and_resize \
                                                                                     --do_shuffle_cams \
                                                                                     --checkpoint_path "/home/mbarin22/projects/simplebevdist/checkpoints/[224, 400]simplebevdist_res101_bs_8x5_lr_0.0003_25k_aug_seed41miou_fixed/best.pt" \