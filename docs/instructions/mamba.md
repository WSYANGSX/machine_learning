Please refer to https://github.com/luluyuu/COMO

# 1. Install dependencies
pip install causal_conv1d-1.1.1+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
# mamba_ssm-1.2.0链接: https://pan.baidu.com/s/16X_vhdSkRB_9y6GKOgv4cw?pwd=1234 提取码: 1234

# 2. Replace mamba_simple.py & local_scan.py
# (copy modified ./ssm/mamba_simple.py and ./ssm/local_scan.py  into the installed mamba-ssm package folder as fig.1 )

# 3. Replace selective_scan_interface.py 
# (copy  modified ./ssm/selective_scan_interface.py into the correct path in mamba-ssm/ops)

mamba-ssm/
├── modules/
│   ├── mamba_simple.py     ← Replace this file (Step 2)
│   └──  local_scan.py       ← Replace this file (Step 2)
├── ops/
│   └── selective_scan_interface.py ← Replace this file (Step 3)

Note: For higher versions of mamba, please modify the code yourself.