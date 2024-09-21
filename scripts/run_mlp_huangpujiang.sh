###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-09-19 16:43:30
### 

# datetime
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo $current_datetime

dataset_path="../datasets/HuangpujiangRiver_check"
result_path="../results"
checkpoint_path="../checkpoints"
train_sample_ratio=0.8
learning_rate=0.001
epochs=100
device="cuda:0"
seed=1
situation="mlp_Huangpujiang, "
corSituation="with cor insider sigmoid function, "
peakPickNum=0

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
    --corSituation $corSituation \
    --peakPickNum $peakPickNum > ../logs/mlp_huangpujiang_${current_datetime}.log 2>&1 &