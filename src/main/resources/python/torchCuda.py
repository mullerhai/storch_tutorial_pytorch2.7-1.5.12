import torch
print(torch.cuda.is_available())
print(f"Version: {torch.__version__}, GPU: {torch.cuda.is_available()}, NUM_GPU: {torch.cuda.device_count()}")

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
# print(torch.cuda.version())