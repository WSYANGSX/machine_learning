#!/bin/bash

# ================= 配置区 =================
# Python解释器的绝对路径
PY_ENV="/home/yangxf/anaconda3/envs/ai/bin/python"

# 获取当前日期时间，用于创建不重复的日志目录
START_TIME=$(date "+%Y%m%d_%H%M%S")
LOG_DIR="logs_${START_TIME}"

# 创建日志文件夹
mkdir -p "$LOG_DIR"
echo "日志将保存在文件夹: $LOG_DIR"
# =========================================

echo "================ 批量任务开始 ================"

# 1. 运行 yolo_v8_ir
echo "[$(date)] 开始运行程序1: yolo_v8_ir.py"
$PY_ENV yolo_v8_ir.py > "$LOG_DIR/1_yolo_v8_ir.log" 2>&1

# 2. 运行 yolo_v8_rgb
echo "[$(date)] 开始运行程序2: yolo_v8_rgb.py"
$PY_ENV yolo_v8_rgb.py > "$LOG_DIR/2_yolo_v8_rgb.log" 2>&1

# 3. 运行 yolo_v13_ir
echo "[$(date)] 开始运行程序3: yolo_v13_ir.py"
$PY_ENV yolo_v13_ir.py > "$LOG_DIR/3_yolo_v13_ir.log" 2>&1

# 4. 运行 yolo_v13_rgb
echo "[$(date)] 开始运行程序4: yolo_v13_rgb.py"
$PY_ENV yolo_v13_rgb.py > "$LOG_DIR/4_yolo_v13_rgb.log" 2>&1

# # 5. 运行 m2i2ha_v8_n
# echo "[$(date)] 开始运行程序5: m2i2ha_v8_n.py"
# $PY_ENV /home/yangxf/WorkSpace/machine_learning/examples/train/detection/m2i2ha/m2i2ha_v8_n.py > "$LOG_DIR/5_m2i2ha_v8_n.log" 2>&1

# # 6. 运行 m2i2ha_v8_s
# echo "[$(date)] 开始运行程序6: m2i2ha_v8_s.py"
# $PY_ENV /home/yangxf/WorkSpace/machine_learning/examples/train/detection/m2i2ha/m2i2ha_v8_s.py > "$LOG_DIR/6_m2i2ha_v8_s.log" 2>&1

# 7. 运行 m2i2ha_v13_n
echo "[$(date)] 开始运行程序7: m2i2ha_v13_n.py"
$PY_ENV /home/yangxf/WorkSpace/machine_learning/examples/train/detection/m2i2ha/m2i2ha_v13_n.py > "$LOG_DIR/7_m2i2ha_v13_n.log" 2>&1

# 8. 运行 m2i2ha_v13_s
echo "[$(date)] 开始运行程序8: m2i2ha_v13_s.py"
$PY_ENV /home/yangxf/WorkSpace/machine_learning/examples/train/detection/m2i2ha/m2i2ha_v13_s.py > "$LOG_DIR/8_m2i2ha_v13_s.log" 2>&1

# # 9. 运行 como (你原脚本最后也是写的程序8，我这里更正为9)
# echo "[$(date)] 开始运行程序9: como.py"
# $PY_ENV /home/yangxf/WorkSpace/machine_learning/examples/train/detection/como/como.py > "$LOG_DIR/9_como.log" 2>&1

echo "================ 所有程序已完成 ================"
echo "请查看 $LOG_DIR 目录下的日志文件以确认结果。"
