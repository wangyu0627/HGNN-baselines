#!/bin/bash

# 循环执行三次
for i in {1..5}
do
    python main_hgmae.py --dataset aminer --task classification --use_cfg --gpu=0
done