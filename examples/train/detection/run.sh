#!/bin/bash

# # 顺序运行两个程序（不是同时运行）
# echo "开始运行程序1..."
# /home/yangxf/anaconda3/envs/ai/bin/python yolo_v8_ir.py

# echo "开始运行程序2..."
# /home/yangxf/anaconda3/envs/ai/bin/python yolo_v8_rgb.py

echo "开始运行程序3..."
/home/yangxf/anaconda3/envs/ai/bin/python yolo_v13_ir.py

echo "开始运行程序4..."
/home/yangxf/anaconda3/envs/ai/bin/python yolo_v13_rgb.py

echo "程序都完成了"
