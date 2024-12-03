#!/bin/bash

echo "Training..."


torchrun --master_port=12345 --nproc_per_node=1 train.py \
                                                      --dataset_path "/datasets/nuscenes/" \
                                                      --batch_size 8 \
                                                      --backbone "dinov2_l" \
                                                      --use_lora \
                                                      --lora_rank 32 \
                                                      --resolution 448 784 \
                                                      --ncams 6 \
                                                      --do_rgbcompress \
                                                      --use_checkpoint \
                                                      --checkpoint_path "checkpoints/[448, 784]simplebev:dinov2_l_bs:8x2_lr:0.001_8k/3.pt" \
                                                      --validate \




# torchrun --master_port=10211 --nproc_per_node=2 "train.py" \
#                                                                                      --dataset_path "/datasets/nuscenes/" \
#                                                                                      --batch_size 8 \
#                                                                                      --backbone "res101" \
#                                                                                      --resolution 448 800 \
#                                                                                      --ncams 6 \
#                                                                                      --do_rgbcompress \
#                                                                                      --gradient_acc_steps 5 \
#                                                                                      --num_steps 25000 \
#                                                                                      --log_freq 5000 \
#                                                                                      --evaluate_all_val \
#                                                                                      --aug     \
#                                                                                      --model_save_path "/home/mbarin22/projects/robustbev/robustbev/checkpoints/" \


# torchrun --master_port=10211 --nproc_per_node=2 "train.py" \
#                                                                                      --dataset_path "/datasets/nuscenes/" \
#                                                                                      --batch_size 16 \
#                                                                                      --backbone "dinov2_b" \
#                                                                                      --use_lora \
#                                                                                      --lora_rank 32 \
#                                                                                      --resolution 224 392 \
#                                                                                      --ncams 6 \
#                                                                                      --do_rgbcompress \
#                                                                                      --gradient_acc_steps 1 \
#                                                                                      --learning_rate 0.001 \
#                                                                                      --num_steps 25000 \
#                                                                                      --log_freq 5000 \
#                                                                                      --evaluate_all_val \
#                                                                                      --aug     \
#                                                                                      --model_save_path "/home/mbarin22/projects/robustbev/robustbev/checkpoints/" \



# torchrun --master_port=1320 --nproc_per_node=4 "train.py" \
#                                                                                      --dataset_path "/datasets/nuscenes/" \
#                                                                                      --batch_size 8 \
#                                                                                      --backbone "dinov2_l" \
#                                                                                      --use_lora \
#                                                                                      --lora_rank 32 \
#                                                                                      --resolution 448 784 \
#                                                                                      --ncams 6 \
#                                                                                      --do_rgbcompress \
#                                                                                      --gradient_acc_steps 5 \
#                                                                                      --learning_rate 0.001 \
#                                                                                      --num_steps 8000 \
#                                                                                      --log_freq 1000 \
#                                                                                      --model_save_path "/home/mbarin22/projects/robustbev/robustbev/checkpoints/" \