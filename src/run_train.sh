#!/bin/bash

#SBATCH --job-name=train_vacnic_ICEK
#SBATCH -o vacnic.out
#SBATCH --error=train_TnT_error.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1000:00:00

source /data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/bin/activate
export PATH='/data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/bin:$PATH'
export PYTHONPATH='/data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/lib/python3.9/site-packages:$PYTHONPATH'
export PATH="/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/Harmful-Videos-Detections/ffmpeg/bin:$PATH"
export JAVA_HOME=/data2/npl/ICEK/Java
export PATH=$JAVA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=5

CUDA_VISIBLE_DEVICES=5 python /data2/npl/ICEK/vacnic/VACNIC-VN/src/official_train_v3_cleaned.py  --num_epoch 2 --out_dir "/data2/npl/ICEK/Image-captioning-for-Vietnamese/output5" --train_batch_size 4