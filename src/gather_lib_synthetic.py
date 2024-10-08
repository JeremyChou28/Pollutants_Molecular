import os
import pandas as pd

def concat_csv_in_subdirs(root_dir, target_file_name, output_file):
    # initialize an empty dataframe
    all_data = pd.DataFrame()

    for subdir, _, files in os.walk(root_dir):
        if target_file_name in files:
            file_path = os.path.join(subdir, target_file_name)
            try:
                data = pd.read_csv(file_path)
                all_data = pd.concat([all_data, data], ignore_index=True)
                print(f"File {file_path} successfully added.")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    all_data.to_csv(output_file, index=False)
    print(f"All files successfully concatenated into {output_file}.")

if __name__ == "__main__":
    root_dir = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/branches/Pollutants_Molecular/datasets"  
    target_file_name = "all_similarity_scores.csv"       
    output_file = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/branches/Pollutants_Molecular/datasets/Synthetic/all_similarity_scores.csv"         

    concat_csv_in_subdirs(root_dir, target_file_name, output_file)
