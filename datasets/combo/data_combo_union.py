import os
import pandas as pd
import shutil

folderPath = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check"
initial_index = 1
folderPath1 = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/ChangjiangRiver_check"
folderPath2 = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/Honey_neg_check"
folderPath3 = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/Honey_pos_check"
folderPath4 = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/HuangpujiangRiver_check"
folderPath5 = "/home/zgjgroup/lyz/hjwrw/MolecularNetwork/datasets/inputfiles_check/Sediment_check"

def process_file(folder_path , index):
    nodes_data = pd.read_csv(folder_path + "/nodes_data.csv")
    nodes_data['newID'] = range(index, index + len(nodes_data))
    edges = pd.read_csv(folder_path + "/edges.csv")
    m = pd.read_csv(folder_path + "/m.csv")
    correlation = pd.read_csv(folder_path + "/cor.csv")
    
    for i, row in nodes_data.iterrows():
        # edges.loc[edges['Library_node'] == row['ID'], 'Library_node'] = row['newID']
        # edges.loc[edges['Connected_node'] == row['ID'], 'Connected_node'] = row['newID']
        # m.loc[m['ID'] == row['ID'], 'ID'] = row['newID']
        # correlation.loc[correlation['Connected_node'] == row['ID'], 'Connected_node'] = row['newID']
        # correlation.loc[correlation['Library_node'] == row['ID'], 'Library_node'] = row['newID']
        edges.replace(row['ID'], row['newID'], inplace=True)
        m.replace(row['ID'], row['newID'], inplace=True)
        correlation.replace(row['ID'], row['newID'], inplace=True)
        old_id = str(row['ID'])
        new_id = str(row['newID'])
        source_folder = folder_path + "/f_filter"
        target_folder = folderPath + "/combo/f_filter"
        for file_name in os.listdir(source_folder):
            # build file name
            old_file_name = f"{old_id}.csv" 
            new_file_name = f"{new_id}.csv" 
    
            # make complete path
            old_file_path = os.path.join(source_folder, old_file_name)
            new_file_path = os.path.join(target_folder, new_file_name)
    
            # do condition to make sure the efficiency of the code
            if os.path.exists(new_file_path):
                break
            elif os.path.exists(old_file_path):
                shutil.copy(old_file_path, new_file_path)
                print(f"Copied and renamed {old_file_name} to {new_file_name}")
            else: 
                break 
    print(edges)
    print(nodes_data)
    print(m)
    print(correlation)
    
    # exchange the position of the columns
    nodes_data[['newID', 'SMILES','ID']]
    nodes_data.drop('ID', axis=1,inplace=True)
    nodes_data.rename(columns={'newID':'ID'}, inplace=True)
    print(nodes_data)
    print(len(nodes_data))
    newIndex = index + len(nodes_data)
    return nodes_data, edges, m, correlation, newIndex

nodes_data1, edges1, m1, correlation1, Index1 = process_file(folderPath1, initial_index)
nodes_data2, edges2, m2, correlation2, Index2 = process_file(folderPath2, Index1)
nodes_data3, edges3, m3, correlation3, Index3 = process_file(folderPath3, Index2)
nodes_data4, edges4, m4, correlation4, Index4 = process_file(folderPath4, Index3)
nodes_data5, edges5, m5, correlation5, Index5 = process_file(folderPath5, Index4)

nodes_data_ = pd.concat([nodes_data1, nodes_data2, nodes_data3, nodes_data4, nodes_data5], ignore_index=True)
edges_ = pd.concat([edges1, edges2, edges3, edges4, edges5], ignore_index=True)
m_ = pd.concat([m1, m2, m3, m4, m5], ignore_index=True)
correlation_ = pd.concat([correlation1, correlation2, correlation3, correlation4, correlation5], ignore_index=True)

nodes_data_.to_csv(folderPath + "/combo/nodes_data.csv", index=False)
m_.to_csv(folderPath + "/combo/m.csv", index=False)
edges_.to_csv(folderPath + "/combo/edges.csv", index=False)
correlation_.to_csv(folderPath + "/combo/cor.csv", index=False)