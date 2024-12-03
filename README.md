# Robust Bird's Eye View Segmentation by Adapting DINOv2

### [Paper](https://www.arxiv.org/pdf/2409.10228) | Webpage (*In Progress*)

This is the official implementation of the paper *Robust Bird's Eye View Segmentation by Adapting DINOv2* presented at ECCV 2024 - 2nd Workshop on Vision-Centric Autonomous Driving.

![](figures/methodology.png)


## Introduction

### 1\. Clone the Repository
```
git clone https://github.com/mrabiabrn/robustbev.git
cd robustbev
```

### 2\. Setup the Environment
Create a Conda environment and install the required dependencies:
```
conda create -n robustbev
conda activate robustbev
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2  pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Download Dataset

Download NuScenes from [this link](https://www.nuscenes.org/) to `root/to/nuscenes`.


### Adaptation Training

- Use DINOv2 ViT-B as backbone with resolution 224x400:
```
torchrun --master_port=12345 --nproc_per_node=<#gpus> train.py \
                                                      --dataset_path "root/to/nuscenes" \
                                                      --batch_size 16 \
                                                      --backbone "dinov2_b" \
                                                      --use_lora \
                                                      --lora_rank 32 \
                                                      --resolution 224 392 \
                                                      --ncams 6 \
                                                      --do_rgbcompress \
                                                      --gradient_acc_steps 1 \
                                                      --learning_rate 0.001 \
                                                      --num_steps 25000 \
                                                      --log_freq 5000 \
                                                      --evaluate_all_val \
                                                      --aug     \
                                                      --model_save_path "root/to/ckpt" \
```
This evaluates to 42.3 .

- Use DINOv2 ViT-L as backbone with resolution 448x784:
```
torchrun --master_port=12345 --nproc_per_node=<#gpus> train.py \
                                                      --dataset_path "root/to/nuscenes" \
                                                      --batch_size 8 \
                                                      --backbone "dinov2_l" \
                                                      --use_lora \
                                                      --lora_rank 32 \
                                                      --resolution 448 784 \
                                                      --ncams 6 \
                                                      --do_rgbcompress \
                                                      --gradient_acc_steps 5 \
                                                      --learning_rate 0.001 \
                                                      --num_steps 8000 \
                                                      --log_freq 1000 \
                                                      --model_save_path "root/to/ckpt" \
```
For full validation set evaluation of models trained with DINOv2 ViT-L backbone, run the inference script after training:
```
torchrun --master_port=12345 --nproc_per_node=1  train.py \
                                                      --dataset_path "root/to/nuscenes" \
                                                      --batch_size 8 \
                                                      --backbone "dinov2_l" \
                                                      --use_lora \
                                                      --lora_rank 32 \
                                                      --resolution 448 784 \
                                                      --ncams 6 \
                                                      --do_rgbcompress \
                                                      --use_checkpoint \
                                                      --checkpoint_path "checkpoints/[448, 784]simplebev:dinov2_l_bs:8x2_lr:0.001_8k/3.pt"
                                                      --validate \
```



### Reproducing SimpleBEV
To reproduce the reported result for SimpleBEV, run the following command:
```
torchrun --master_port=12345 --nproc_per_node=<#gpus> train.py \
                                                      --dataset_path "root/to/nuscenes" \
                                                        --batch_size 8 \
                                                        --backbone "res101" \
                                                        --resolution 224 400 \
                                                        --ncams 6 \
                                                        --do_rgbcompress \
                                                        --gradient_acc_steps 5 \
                                                        --num_steps 25000 \
                                                        --rand_flip \
                                                        --rand_crop_and_resize \
                                                        --do_shuffle_cams \
                                                        --log_freq 5000 \
                                                        --model_save_path "root/to/ckpt" \

```
At the end of the training, you should get mIoU of **42.3**. You can also increase the resolution for reproducing the results with original resolution (47.4)


### Robustness Analysis

```
cd robustness
torchrun --master_port=12345 --nproc_per_node=1 robustness.py \
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
                                                                                     --checkpoint_path "root/to/ckpt" \

```
## Citation

If you use this code in your research, please cite the following:
```bibtex
@article{barin2024robust,
  title={Robust Bird's Eye View Segmentation by Adapting DINOv2},
  author={Bar{\i}n, Merve Rabia and Aydemir, G{\"o}rkay and G{\"u}ney, Fatma},
  journal={arXiv preprint arXiv:2409.10228},
  year={2024}
}
```

## Acknowledgments
This repository incorporates code from several public works, including [SimpleBEV](https://github.com/aharley/simple_bev), [RoboBEV](https://github.com/Daniel-xsy/RoboBEV), [MeLo](https://github.com/JamesQFreeman/LoRA-ViT), and [SOLV](https://github.com/gorkaydemir/SOLV). Special thanks to the authors of these projects for making their code available.