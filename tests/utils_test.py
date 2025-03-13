import os
import sys
from machine_learning.utils import print_dict


if __name__ == "__main__":
    # a = {
    #     "training": {"batch_size": 256, "epochs": 200, "grad_clip": 5.0, "save_interval": 10},
    #     "optimizer": {
    #         "type": "Adam",
    #         "learning_rate": 0.001,
    #         "beta1": 0.9,
    #         "beta2": 0.999,
    #         "eps": 1e-08,
    #         "weight_decay": 1e-05,
    #         "scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 5},
    #     },
    #     "data": {
    #         "data_path": "./Machine learning/Auto_Encoder/data",
    #         "num_workers": 4,
    #         "norm_mean": 0.1307,
    #         "norm_std": 0.3081,
    #     },
    #     "model": {"initialize_weights": True},
    #     "logging": {
    #         "log_interval": 10,
    #         "log_dir": "./Machine learning/Auto_Encoder/logs",
    #         "model_dir": "./Machine learning/Auto_Encoder/checkpoints",
    #     },
    # }

    # 读取所有环境变量
    path = sys.path
    print(path)

    # 获取当前工作目录
    a = {
        "type": "Adam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-08,
        "weight_decay": 1e-05,
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 5,
            "logging": {
                "log_interval": 10,
                "log_dir": "./Machine learning/Auto_Encoder/logs",
                "model_dir": "./Machine learning/Auto_Encoder/checkpoints",
            },
        },
        "abc": {
            "type": "ReduceLROnPlateau",
            "factor": 0.5,
            "patience": 5,
            "logging": {
                "log_interval": 10,
                "log_dir": "./Machine learning/Auto_Encoder/logs",
                "model_dir": "./Machine learning/Auto_Encoder/checkpoints",
            },
        },
    }

    print_dict(a)
