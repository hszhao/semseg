import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
cpu_path = os.path.join(cwd, 'cpu')
gpu_path = os.path.join(cwd, 'gpu')

cpu = load('psamask_cpu', [
    os.path.join(cpu_path, 'operator.cpp'),
    os.path.join(cpu_path, 'psamask.cpp'),
], build_directory=cpu_path, verbose=False)

if torch.cuda.is_available():
    gpu = load('psamask_gpu', [
        os.path.join(gpu_path, 'operator.cpp'),
        os.path.join(gpu_path, 'psamask_cuda.cu'),
    ], build_directory=gpu_path, verbose=False)