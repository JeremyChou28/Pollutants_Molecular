###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-10-10 17:01:04
### 
cd ../src
# python taniLibGenerator.py \
# --dataset_path "../datasets/ChangjiangRiver_check" \
# --result_path "../datasets/ChangjiangRiver_check/train_tani_lib"

# python -u taniLibGenerator.py \
# --dataset_path "../datasets/Honey_neg_check" \
# --result_path "../datasets/Honey_neg_check/train_tani_lib"

# python -u taniLibGenerator.py \
# --dataset_path "../datasets/Honey_pos_check" \
# --result_path "../datasets/Honey_pos_check/train_tani_lib"

# python -u taniLibGenerator.py \
# --dataset_path "../datasets/HuangpujiangRiver_check" \
# --result_path "../datasets/HuangpujiangRiver_check/train_tani_lib"

# python -u taniLibGenerator.py \
# --dataset_path "../datasets/Sediment_check" \
# --result_path "../datasets/Sediment_check/train_tani_lib"

python -u taniLibGenerator.py \
--dataset_path "../datasets/Synthetic" \
--result_path "../datasets/Synthetic/train_tani_lib"