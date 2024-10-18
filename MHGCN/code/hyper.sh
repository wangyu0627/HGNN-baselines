#!/bin/bash
# 设置总迭代次数
total_iterations=$((100))
# 循环lr
for lr in $(seq 0.01 -0.0001 0.0001)
do
    progress=$((progress + 1))
    printf "[%-100s] %d%%\r" $(printf "#%.0s" $(seq 1 $((progress * 100 / total_iterations)))) $((progress * 100 / total_iterations))
    echo "Running main_MHGCN.py with lr=$lr"
    python main_MHGCN.py --lr $lr
done

echo -e "\nLoop completed!"










