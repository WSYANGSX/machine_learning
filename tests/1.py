import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建简单数据集
data = torch.arange(10)
dataset = TensorDataset(data)
loader = DataLoader(dataset, batch_size=3, shuffle=True)

# print("同一个DataLoader，多个epoch：")
# for epoch in range(3):
#     print(f"Epoch {epoch}: ", end="")
#     batches = []
#     for batch in loader:
#         print(id(loader))
#         batches.append(batch[0].tolist())
#     print(batches)

print("\n创建新的DataLoader：")
for epoch in range(3):
    new_loader = DataLoader(dataset, batch_size=3, shuffle=True)  # 每次都新建
    batches = []
    for batch in new_loader:
        print(id(new_loader))
        batches.append(batch[0].tolist())
    print(f"Epoch {epoch}: {batches}")
