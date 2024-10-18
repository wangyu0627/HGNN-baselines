#!/bin/bash

# 设置初始lr
lr=0.01

# 循环运行main_han.py，每次将lr减半
for i in {1..100}
do
    echo "Running main_han.py with lr=$lr"
    python main_han.py --lr $lr

    # 将lr减半
    lr=$(awk "BEGIN {print $lr-0.0001}")
done
