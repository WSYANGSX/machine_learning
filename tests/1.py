from pathlib import Path

# 指定模型目录路径
model_dir = Path("/home/yangxf/WorkSpace/machine_learning/runs/unet/unet_car_2026-04-07_10-52/ckpt")

# 找出所有 .pth 文件并按 epoch 数字排序
pth_files = list(model_dir.glob("checkpoint_epoch_*.pth"))

if pth_files:
    # 提取 epoch 数字并排序
    epoch_numbers = []
    for f in pth_files:
        # 从 "checkpoint_epoch_10.pth" 中提取 10
        epoch_num = int(f.stem.split("_")[-1])
        epoch_numbers.append((epoch_num, f))

    # 按 epoch 数字排序，取最大的
    epoch_numbers.sort(key=lambda x: x[0])
    last_model = epoch_numbers[-1][1]

    print(f"最后一个模型: {last_model.name}")
    print(f"完整路径: {last_model}")
    print(f"Epoch: {epoch_numbers[-1][0]}")
else:
    print("未找到模型文件")
