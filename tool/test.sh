#!/bin/sh

## uncomment for slurm
##SBATCH -p gpu
##SBATCH --gres=gpu:1
##SBATCH -c 10
pip install tensorboardX

PYTHON=python

dataset=$1
exp_name=$2
exp_dir=/logs/exp/${dataset}/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
cp tool/test.sh tool/test.py ${config} ${exp_dir}

export PYTHONPATH=$PYTHONPATH:/workspace:./
$PYTHON -u ${exp_dir}/test.py \
  --config=${config} \
  2>&1 | tee ${result_dir}/test-$now.log
