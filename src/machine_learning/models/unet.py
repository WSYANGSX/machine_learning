import torch.nn as nn

from machine_learning.models import BaseNet


class UNet(BaseNet):
    def __init__(self):
        """扩散模型中UNet网络实现

        Args:
            cfg (str): 算法配置, YAML文件路径.
            models (Mapping[str, BaseNet]): 算法所需的网络模型.
            name (str | None, optional): 算法名称. Defaults to None.
            device (Literal[&quot;cuda&quot;, &quot;cpu&quot;, &quot;auto&quot;], optional): 算法运行设备. Defaults to "auto".
        """
        super().__init__()
