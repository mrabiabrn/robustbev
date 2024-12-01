#!/bin/bash

echo "Training..."

torchrun --master_port=1325 --nproc_per_node=2 "train.py" \
                                                                                     --dataset_path "/datasets/nuscenes/" \
                                                                                     --batch_size 16 \
                                                                                     --backbone "dinov2_b" \
                                                                                     --use_lora \
                                                                                     --lora_rank 32 \
                                                                                     --resolution 224 392 \
                                                                                     --ncams 6 \
                                                                                     --do_rgbcompress \
                                                                                     --gradient_acc_steps 1 \
                                                                                     --num_steps 25000 \
                                                                                     --rand_flip \
                                                                                     --rand_crop_and_resize \
                                                                                     --do_shuffle_cams \
                                                                                     --log_freq 5000 \
                                                                                     --model_save_path "/home/mbarin22/projects/robustbev/robustbev/checkpoints/" \