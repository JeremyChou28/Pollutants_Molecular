###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-09-19 16:43:30
### 

# datetime
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo $current_datetime

dataset_path="../datasets/combo"
result_path="../results/combo"
checkpoint_path="../checkpoints/combo"
train_sample_ratio=0.8
learning_rate=0.001
epochs=100
device="cuda:0"
seed=1
situation="mlp_Combo, "
# cor_situation="with_cor_outside_the_sigmoid_function, "
cor_situation="without_cor"
peak_pick_num=10

nohup python -u ../src/main.py \
    --dataset_path $dataset_path \
    --result_path $result_path \
    --checkpoint_path $checkpoint_path \
    --train_sample_ratio $train_sample_ratio \
    --learning_rate $learning_rate \
    --epochs $epochs \
    --device $device \
    --seed $seed \
    --situation $situation \
    --cor_situation $cor_situation \
    --peak_pick_num $peak_pick_num > ../logs/mlp_combo_${current_datetime}.log 2>&1 &