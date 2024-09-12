'''
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2024-09-12 15:46:56
'''
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import rdMolDescriptors
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")


def generate_t_score(args):
    # 设置输入和输出目录
    input_dir = args.input_file_path
    output_dir = args.output_file_path
    os.makedirs(output_dir, exist_ok=True)

    # 读取目录中的所有CSV文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # 处理每个文件
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        data = pd.read_csv(file_path)
        
        # 确保数据中至少有一个化合物，以免出现错误
        if data.shape[0] < 2:
            print(f"Not enough data in {file_path} for analysis. It needs at least 2 compounds.")
            continue  # 跳过当前文件的进一步处理
        
        # 解析SMILES字符串
        mols = [Chem.MolFromSmiles(smiles) for smiles in data['SMILES']]
        
        # 计算化学指纹，并确保所有指纹都成功生成
        # fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols if mol is not None]
        generator = rdMolDescriptors.GetMorganFingerprintAsBitVect
        fps = [generator(mol, radius=2, nBits=1024) for mol in mols if mol is not None]
        valid_names = data.loc[[mol is not None for mol in mols], 'SMILES']
        
        if len(fps) < 2:
            print(f"Not enough valid fingerprints in {file_path} for similarity calculation. It needs at least 2 valid compounds.")
            continue  # 跳过当前文件的进一步处理
        
        # 计算指纹相似度矩阵
        n = len(fps)
        fp_sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                fp_sim[i, j] = fp_sim[j, i] = sim
        
        # 提取第一个物质与其它所有物质的相似度
        first_compound_name = valid_names.iloc[0]
        other_compounds_names = valid_names.iloc[1:]
        similarity_scores = fp_sim[0, 1:]
        
        # 创建输出数据表
        output_data = pd.DataFrame({
            'ID1': [first_compound_name] * len(other_compounds_names),
            'ID2': other_compounds_names,
            'Similarity': similarity_scores
        })
        
        # 保存结果到CSV
        output_data.to_csv(os.path.join(output_dir, file_name), index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    generate_t_score(args)