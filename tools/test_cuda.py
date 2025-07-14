import torch

print("CUDA 是否可用:", torch.cuda.is_available())
print("CUDA 设备数量:", torch.cuda.device_count())
print("CUDA 设备名称:", torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    print("GPU is available and will be used!")
else:
    print("GPU is not available, using CPU instead.")
