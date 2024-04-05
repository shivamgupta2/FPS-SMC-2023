#/bin/bash

# $1: task
# $2: gpu number
# $3: output directory
# $4: c_rate, default 0.95
# $5: particle_size, default 5

python3 sample_condition.py \
    --model_config=configs/model_imagenet_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/inpainting_imagenet_config.yaml \
    --particle_size=5;
