import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import timm

m = timm.create_model("mobilenetv3_large_100", pretrained=True)
m.eval()

pretrained_cfg = m.pretrained_cfg
print(pretrained_cfg)

data_cfg = timm.data.resolve_data_config(pretrained_cfg)
print(data_cfg)

transform = timm.data.create_transform(**data_cfg)
print(transform)