###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-10-10 17:01:04
### 
cd ../src
python taniLibGenerator.py \
--input_file_path "../datasets/ChangjiangRiver_check/library&librarymatch" \
--output_file_path "../datasets/ChangjiangRiver_check/tani_lib"

# python -u taniLibGenerator.py \
# --input_file_path "../datasets/Honey_neg_check/library&librarymatch" \
# --output_file_path "../datasets/Honey_neg_check/tani_lib"

# python -u taniLibGenerator.py \
# --input_file_path "../datasets/Honey_pos_check/library&librarymatch" \
# --output_file_path "../datasets/Honey_pos_check/tani_lib"

# python -u taniLibGenerator.py \
# --input_file_path "../datasets/HuangpujiangRiver_check/library&librarymatch" \
# --output_file_path "../datasets/HuangpujiangRiver_check/tani_lib"

# python -u taniLibGenerator.py \
# --input_file_path "../datasets/Sediment_check/library&librarymatch" \
# --output_file_path "../datasets/Sediment_check/tani_lib"