#!/bin/bash
# 设置总迭代次数
total_iterations=$((100 * 6 * 3 * 2))  # 100 lr * 6 dropout * 5 num_layer * 4 num_heads
# 循环lr
for lr in $(seq 0.01 -0.0001 0.0001)
do
    # 循环dropout
    for dropout in $(seq 0.0 0.1 0.5)
    do
        # 循环num_layer
        for num_layer in 1 2 4
        do
            # 循环num_heads
            for num_heads in 2 4
            do
                progress=$((progress + 1))
                printf "[%-100s] %d%%\r" $(printf "#%.0s" $(seq 1 $((progress * 100 / total_iterations)))) $((progress * 100 / total_iterations))
                echo "Running main_hgt.py with lr=$lr, dropout=$dropout, num_layer=$num_layer, num_heads=$num_heads"
                python main_hgt.py --lr $lr --dropout $dropout --num_layer $num_layer --num_heads $num_heads
            done
        done
    done
done

echo -e "\nLoop completed!"










