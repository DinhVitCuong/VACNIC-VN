#!/bin/bash

#SBATCH --job-name=inference
#SBATCH -o /data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/implicit-hatespeech-detection/script/inference_gemma2.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1000:00:00

source /data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/bin/activate
export PATH='/data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/bin:$PATH'
export PYTHONPATH='/data/npl/ICEK/Image-captioning-for-Vietnamese/vacnic/lib/python3.9/site-packages:$PYTHONPATH'

python '/data/npl/ICEK/Image-captioning-for-Vietnamese/process data/add_faces_objects.py'