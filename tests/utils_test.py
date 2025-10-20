from tqdm import trange
import time

# 使用多个参数自定义进度条
for i in trange(
    10,
    desc="主要进度",
    unit="文件",
    unit_scale=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
):
    # 可以动态更新描述
    if i % 2 == 0:
        tqdm_instance.set_description(f"处理偶数项 {i}")
    else:
        tqdm_instance.set_description(f"处理奇数项 {i}")

    time.sleep(0.2)
