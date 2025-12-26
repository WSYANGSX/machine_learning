#!/bin/bash

# 顺序运行两个程序（不是同时运行）
echo "开始运行程序1..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_n.py

echo "开始运行程序2..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_s.py

echo "两个程序都完成了"
