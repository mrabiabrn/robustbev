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
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 3\. Download Dataset

Download NuScenes from [this link](https://www.nuscenes.org/) to `root/to/nuscenes`.


### 4\. Adaptation Training

We provide example commands below, you can play with the arguments to reproduce our experiments.  

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
This evaluates to *42.3* in our environment.

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

### 5\. Inference

We log the performance of random validation subset during trainings with DINOv2 ViT-L due to time complexity. 
For full validation set evaluation of models trained with DINOv2 ViT-L backbone, run the inference script after training:
```
torchrun --master_port=12345 --nproc_per_node=1  train.py \
                                                      --dataset_path "root/to/nuscenes" \
                                                      --batch_size 1 \
                                                      --backbone "dinov2_l" \
                                                      --use_lora \
                                                      --lora_rank 32 \
                                                      --resolution 448 784 \
                                                      --ncams 6 \
                                                      --do_rgbcompress \
                                                      --use_checkpoint \
                                                      --checkpoint_path "root/to/ckpt"
                                                      --validate \
```
This evaluates to *48.3*. We got better results than the reported results using this repo.

### 6\. (Optional) Reproducing SimpleBEV 
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
                                                        --aug \
                                                        --log_freq 5000 \
                                                        --evaluate_all_val \
                                                        --model_save_path "root/to/ckpt" \

```
At the end of the training, you should get mIoU of *42.3*. You can also increase the resolution for reproducing the results with its original resolution which is *47.4*.


### Robustness Analysis

```
cd robustness
torchrun --master_port=12345 --nproc_per_node=1 robustness.py \
                                                                                     --dataset_path "/datasets/nuscenes/" \
                                                                                     --batch_size 1 \
                                                                                     --backbone "res101" \
                                                                                     --resolution 224 400 \
                                                                                     --ncams 6 \
                                                                                     --do_rgbcompress \
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