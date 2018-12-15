from io import BytesIO
import pandas as pd
import torch
import subprocess
import sys


USE_GPU = torch.cuda.is_available()


def select_device():
    """Selects GPU with the most available memory or CPU if cuda is not enabled."""
    if USE_GPU:
        try:
            gpus = subprocess.check_output(['nvidia-smi', '--format=csv', '--query-gpu=memory.used,memory.free'])
            stream = BytesIO(gpus)
            dataframe = pd.read_csv(stream, names=['memory.used', 'memory.free'], skiprows=1)
            dataframe['memory.free'] = dataframe['memory.free'].map(lambda x: float(x.rstrip(' [MiB]')))
            device_idx = dataframe['memory.free'].idxmax()
            device = torch.device(f'cuda:{device_idx}')
        except Exception:
            device_idx = -1
            device = torch.device('cpu')
    else:
        device_idx = -1
        device = torch.device('cpu')
    return USE_GPU, device, device_idx


def with_cpu(x):
    if USE_GPU:
        return x.cpu()
    else:
        return x


def with_gpu(x):
    if USE_GPU:
        return x.cuda()
    else:
        return x
