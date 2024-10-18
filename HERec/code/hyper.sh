#!/bin/bash
# 设置总迭代次数
total_iterations=$((50))
# 循环lr
for lr in $(seq 0.1 -0.001 0.001)
  do
      progress=$((progress + 1))
      printf "[%-100s] %d%%\r" $(printf "#%.0s" $(seq 1 $((progress * 100 / total_iterations)))) $((progress * 100 / total_iterations))
      echo "Running main_HERec.py with lr=$lr"
      python main_HERec.py --lr $lr

  done

echo -e "\nLoop completed!"