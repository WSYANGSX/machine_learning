#!/bin/bash

# 顺序运行两个程序（不是同时运行）
echo "开始运行程序1..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_v13_n.py > log_m2i2ha_v13_n.txt 2>&1   

echo "开始运行程序2..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_v13_s.py > log_m2i2ha_v13_s.txt 2>&1
 
echo "两个程序都完成了"
