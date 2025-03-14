from machine_learning.utils import cal_conv_output_size
import torch.nn as nn

input_size = (20, 3, 28, 28, 28)
a = nn.Conv3d(in_channels=3, out_channels=5, kernel_size=(4, 2, 5), padding=(2, 1, 3), stride=(3, 4, 1))
b = cal_conv_output_size(input_size, a)
print(b)
