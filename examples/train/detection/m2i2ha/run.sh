# 顺序运行两个程序（不是同时运行）
echo "开始运行程序1..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_v8_n.py

echo "开始运行程序2..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_v8_s.py

echo "开始运行程序3..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_v13_n.py

echo "开始运行程序4..."
/home/yangxf/anaconda3/envs/ai/bin/python m2i2ha_v13_s.py

echo "程序都完成了"