###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-09-19 16:43:30
### 

# datetime
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo $current_datetime

dataset_path="../datasets/Honey_pos_check"
result_path="../results/honey_pos"
checkpoint_path="../checkpoints/honey_pos"
train_sample_ratio=0.8
learning_rate=0.001
epochs=5
device="cuda:0"
seed=1
method_variant="wo_cor"  # "wo_cor", "cor_inside", "cor_outside"
peak_pick_num=10    # 1 5 10 20
scratch=False

nohup python -u ../src/main2.py \
    --scratch $scratch \
    --dataset_path $dataset_path \
    --result_path $result_path \
    --checkpoint_path $checkpoint_path \
    --train_sample_ratio $train_sample_ratio \
    --learning_rate $learning_rate \
    --epochs $epochs \
    --device $device \
    --seed $seed \
    --method_variant $method_variant \
    --peak_pick_num $peak_pick_num > ../logs/Honey_pos_${method_variant}_${current_datetime}.log 2>&1 &