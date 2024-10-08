import os
import pandas as pd


folder_path = '../datasets/Sediment_check/tani_lib'

output_file = '../datasets/Sediment_check/all_similarity_scores.csv'


combined_df = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.to_csv(output_file, index=False)

print(f' All csv have been successfully merged into {output_file}')
