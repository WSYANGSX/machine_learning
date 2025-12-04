import torch
from ultralytics.nn.modules import A2C2f

model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
x = torch.randn(2, 64, 128, 128)
output = model(x)
print(output.shape)
