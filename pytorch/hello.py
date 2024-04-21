import torch
import time

print(f'torch version:{torch.__version__}')

a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

start = time.time()
c = torch.matmul(a, b)
end = time.time()
print(f'{a.device}, time collapsed:{end - start}', c.norm(2))




