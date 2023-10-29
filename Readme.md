# SmokeDiff: Denoising Diffusion Model for Smoke Segmentation
The code is based on [MedSegDiff](https://github.com/WuJunde/MedSegDiff)
## A Quick Overview
![enter image description here](https://i.ibb.co/8KjCF2g/Smoke-Segmentation.png%22%20alt=%22Smoke-Segmentation%22%20border=%220%22%3E)
> We aim to utilize the capability of diffusion models for the smoke segmentation task to segment and capture the elusive boundaries of smoke. The method using denoising networks with U-Net as the main architecture, processing the segmentation map as input and the original image as a condition. Our experimental results on SMOKE5K show that our approach produces competitive visual results.

## Installation
Requirement installation: `pip install -r requirement.txt`

## Usage

**Training**
python scripts/segmentation_training.py --data_dir dataset/train --out_dir output/ --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 8

**Sampling**
python scripts/segmentation_sample.py --data_dir dataset/test --out_dir output/ --model_path output/model.pth --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False

**Evaluation**
python scripts/segmentation_eval.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*

## Model Checkpoint
The checkpoint of the model use in this work can be downloaded here:
[Google drive](https://drive.google.com/file/d/19WialpMTnu7qNNQ16rdwh5bam6_-i674/view?usp=sharing)

## Visualization
![enter image description here](https://i.ibb.co/ctVhrxL/Screenshot-2023-10-23-154357.png%22%20alt=%22Screenshot-2023-10-23-154357%22%20border=%220%22)
![enter image description here](https://i.ibb.co/bg2gb8k/Screenshot-2023-10-23-154237.png%22%20alt=%22Screenshot-2023-10-23-154237%22%20border=%220%22)
![enter image description here](https://i.ibb.co/kKRXytF/Screenshot-2023-10-23-154506.png%22%20alt=%22Screenshot-2023-10-23-154506%22%20border=%220%22)
## Acknowledgement

[tomeramit/SegDiff](https://github.com/tomeramit/SegDiff), [WuJunde/MedSegDiff](https://github.com/WuJunde/MedSegDiff), [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), [openai/guided-diffusion](https://github.com/openai/guided-diffusion), [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), [openai/improved-diffusion](https://github.com/openai/improved-diffusion)
