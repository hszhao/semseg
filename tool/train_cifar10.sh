#!/bin/sh

pip install tensorboardX fire

PYTHON=python

result_dir=/cifar10
sparsity=$1
mkdir -p ${result_dir}
cp tool/train_cifar10.py ${result_dir}/train_${sparsity}.py

export PYTHONPATH=$PYTHONPATH:/workspace:./
$PYTHON -m tool.train_cifar10 --sparsity ${sparsity} --save-dir ${result_dir}
