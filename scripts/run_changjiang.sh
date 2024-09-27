###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-09-19 16:43:30
### 

# datetime
current_datetime=$(date +'%Y-%m-%d-%H-%M-%S')
echo $current_datetime

dataset_path="../datasets/ChangjiangRiver_check"
result_path="../results"
checkpoint_path="../checkpoints"
train_sample_ratio=0.8
learning_rate=0.001
epochs=5
device="cuda:0"
seed=1
method_variant="cor_our"  # "wo_cor", "cor_inside", "cor_outside", "cor_our"
peak_pick_num=10    # 1 5 10 20
scratch=False   # True: Train from scratch; False: Load pre-trained model

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
    --peak_pick_num $peak_pick_num > ../logs/ChangjiangRiver_${method_variant}_${current_datetime}.log 2>&1 &


# # different method
# method_variant_list=("wo_cor" "cor_inside" "cor_outside")
# for method_variant in ${method_variant_list[@]}; do
#     echo "method_variant:" $method_variant
#     nohup python -u ../src/main2.py \
#         --scratch $scratch \
#         --dataset_path $dataset_path \
#         --result_path $result_path \
#         --checkpoint_path $checkpoint_path \
#         --train_sample_ratio $train_sample_ratio \
#         --learning_rate $learning_rate \
#         --epochs $epochs \
#         --device $device \
#         --seed $seed \
#         --method_variant $method_variant \
#         --peak_pick_num $peak_pick_num > ../logs/ChangjiangRiver_${method_variant}_${current_datetime}.log 2>&1 &
#     wait
# done


# # different evaluation metric: top_k
# peak_pick_num_list=(1 5 10 20)
# for peak_pick_num in ${peak_pick_num_list[@]}; do
#     echo "peak_pick_num:" $peak_pick_num
#     nohup python -u ../src/main2.py \
#         --scratch $scratch \
#         --dataset_path $dataset_path \
#         --result_path $result_path \
#         --checkpoint_path $checkpoint_path \
#         --train_sample_ratio $train_sample_ratio \
#         --learning_rate $learning_rate \
#         --epochs $epochs \
#         --device $device \
#         --seed $seed \
#         --method_variant $method_variant \
#         --peak_pick_num $peak_pick_num > ../logs/ChangjiangRiver_${method_variant}_${current_datetime}.log 2>&1 &
#     wait
# done