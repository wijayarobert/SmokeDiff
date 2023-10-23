# SmokeDiff: Denoising Diffusion Model for Smoke Segmentation

## A Quick Overview
![enter image description here](https://i.ibb.co/8KjCF2g/Smoke-Segmentation.png%22%20alt=%22Smoke-Segmentation%22%20border=%220%22%3E)
Our approach refines the diffusion model by modifying the step estimation function based on input information, which merges insights from both the current estimate and the input image. In traditional diffusion models, this function is often depicted using a structure similar to that proposed by Ronneberger et al. in 2015. In our method, the two main components of this structure are labeled as the encoder and decoder. We also introduce two unique encoders that process layers of data. One encoder deals with the input image while the other is designed to handle the segmentation map for the current step.

## Installation
Requirement installation: `pip install -r requirement.txt`

## Usage

**Training**
python scripts/segmentation_training.py --data_dir dataset/train --out_dir output/ --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 8

**Sampling**
python scripts/segmentation_sample.py --data_dir dataset/test --out_dir output/ --model_path output/model.pth --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False

**Evaluation**
python scripts/segmentation_eval.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*

## Visualization
![enter image description here](https://i.ibb.co/ctVhrxL/Screenshot-2023-10-23-154357.png%22%20alt=%22Screenshot-2023-10-23-154357%22%20border=%220%22)
![enter image description here](https://i.ibb.co/bg2gb8k/Screenshot-2023-10-23-154237.png%22%20alt=%22Screenshot-2023-10-23-154237%22%20border=%220%22)
![enter image description here](https://i.ibb.co/kKRXytF/Screenshot-2023-10-23-154506.png%22%20alt=%22Screenshot-2023-10-23-154506%22%20border=%220%22)
## Acknowledgement

[tomeramit/SegDiff](https://github.com/tomeramit/SegDiff), [WuJunde/MedSegDiff](https://github.com/WuJunde/MedSegDiff), [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), [openai/guided-diffusion](https://github.com/openai/guided-diffusion), [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
