import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import warnings
import time
import sys
warnings.filterwarnings("ignore", category=UserWarning)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import train_test_split

sys.path.append("../")
from utils.utils import seed_torch

def data_preprocessing(
    nodeId, smiles, edges, libraryNodes, library_m, correlation, nodeFolder
):
    connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)

    node_df = loadNodeCsv(nodeId, nodeFolder)
    labels = node_df["SMILES"]
    ground_truth_index = labels[labels == smiles].index[0]
    ground_truth_vector = oneHotVect(labels, ground_truth_index)

    # get tanimoto for each of the connected nodes
    tani_list = []
    for smile in node_df["SMILES"].values:
        connectedSmiles = findConnectedSmiles(connected_node_ids, libraryNodes)
        tani = calcTanimotoCoef(smile, connectedSmiles)
        tani_list.append(tani)
    tani = np.array(tani_list)

    connected_lib = findConnectedLibrary(nodeId, edges, libraryNodes)
    m = getMVal(connected_lib, library_m)

    cors = getCorrelation(nodeId, connected_node_ids, correlation)
    return node_df, tani, m, cors, ground_truth_vector, labels



# Helper function: Check the existence of files
def check_files_exist(df, column_name, folder_path):
    missing_files = []
    for item in df[column_name]:
        file_path = os.path.join(folder_path, f"{item}.csv")
        if not os.path.isfile(file_path):
            missing_files.append(item)
    return missing_files


# Helper function: Load node CSV files
def loadNodeCsv(nodeId, folderPath):
    filePath = os.path.join(folderPath, f"{nodeId}.csv")
    if os.path.exists(filePath):
        return pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"No CSV file found for node ID {nodeId} at {filePath}")


# Helper function: Convert SMILES to fingerprint
def smiles2Fingerprint(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


# Helper function: Calculate Tanimoto Coefficient
def calcTanimotoCoef(smile, smileSet):
    smile_fp = smiles2Fingerprint(smile)
    if smile_fp is None:
        return [0] * len(smileSet)

    tanimoto_scores = []
    for s in smileSet:
        s_fp = smiles2Fingerprint(s)
        if s_fp is None:
            tanimoto_scores.append(0)
        else:
            tanimoto_scores.append(TanimotoSimilarity(smile_fp, s_fp))
    return tanimoto_scores

def getTanimotoCoef(smile, smileSet, tani_library):
    tanimoto_scores = []
    
    for smile_ in smileSet:
        condition1 = (tani_library["ID1"] == smile) & (tani_library["ID2"] == smile_)
        condition2 = (tani_library["ID1"] == smile_) & (tani_library["ID2"] == smile)
        tani_ = tani_library.loc[condition1 | condition2, "Similarity"]
        if not tani_.empty:
            tanimoto_scores.append(tani_.values[0])
        else:
            tanimoto_scores.append(0)
    
    return tanimoto_scores
        
# Helper function: Find connected library nodes
def findConnectedLibrary(nodeId, edges, libraryNodes):
    connectedNodes = []
    connectedNodes.extend(edges[edges["Connected_node"] == nodeId]["Library_node"])
    connectedNodes.extend(edges[edges["Library_node"] == nodeId]["Connected_node"])
    connectedDf = pd.Series(connectedNodes)
    connectedLibrary = connectedDf[connectedDf.isin(libraryNodes["ID"])].tolist()
    return connectedLibrary


# Helper function: Find connected SMILES
def findConnectedSmiles(connectedNodes, libraryNodes):
    smileSet = libraryNodes[libraryNodes["ID"].isin(connectedNodes)]["SMILES"].tolist()
    return smileSet


# Helper function: Get correlation values
def getCorrelation(nodeId, nodeList, correlation):
    result = []
    for node in nodeList:
        match = correlation[
            ((correlation["Library_node"] == nodeId) & (correlation["Connected_node"]))
            | (
                (correlation["Library_node"] == node)
                & (correlation["Connected_node"] == nodeId)
            )
        ]

        if not match.empty:
            corVal = match["Cor"].values[0]
        else:
            corVal = 0
        result.append(corVal)
    return result


# Helper function: Get m values
def getMVal(idList, df):
    mVals = []
    for item in idList:
        if item in df["ID"].values:
            mVal = df[df["ID"] == item]["m"].values[0]
        else:
            continue
        mVals.append(mVal)
    return np.array(mVals)


# Helper function: Create one-hot vector for cross entropy
def oneHotVect(labels, index):
    vector = torch.zeros(len(labels), dtype=torch.float32)
    vector[index] = 1.0
    return vector

import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class ScoringDataset(Dataset):
    def __init__(self, f_dir, libraryNodes, library_nodes, final, NlibraryNodes, edges, correlation, library_m, tani_library):
        # 检测是否有可用的 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.data_dir = f_dir
        self.libraryNodes = libraryNodes
        self.library_nodes = library_nodes
        self.final = final
        self.non_library_nodes = NlibraryNodes
        self.edges = edges
        self.correlation = correlation
        self.library_m = library_m
        self.tani_library = tani_library

        # 预加载所有数据并将其转移到 GPU 或 CPU
        self.data_cache = self.preload_all_data()

    def preload_all_data(self):
        """预处理和缓存所有节点的数据并转移到 GPU(如果可用)"""
        data_cache = {}
        for idx in range(len(self.final)):
            node_id = self.final.loc[idx, 'ID']
            smiles = self.final.loc[idx, 'SMILES']

            # 加载CSV文件并提取特征
            node_df = loadNodeCsv(node_id, self.data_dir)
            labels = node_df["SMILES"]
            ground_truth_index = labels[labels == smiles].index[0]
            ground_truth_vector = oneHotVect(labels, ground_truth_index)

            # 获取连接的节点和相关属性
            connected_node_ids = findConnectedLibrary(node_id, self.edges, self.libraryNodes)
            connected_smiles = findConnectedSmiles(connected_node_ids, self.libraryNodes)

            # 获取 f 值并转移到设备
            f = torch.tensor(node_df["Score"], dtype=torch.float32).to(self.device)

            # 计算并存储 Tanimoto 系数
            tani_list = []
            for smile in node_df["SMILES"].values:
                tani = getTanimotoCoef(smile, connected_smiles, self.tani_library)
                # tani = calcTanimotoCoef(smile, connected_smiles)
                tani_list.append(tani)
            tani = torch.tensor(tani_list, dtype=torch.float32).to(self.device)

            # 获取 m 值并转移到设备
            m = torch.tensor(getMVal(connected_node_ids, self.library_m), dtype=torch.float32).to(self.device)

            # 获取相关系数并转移到设备
            cors = torch.tensor(getCorrelation(node_id, connected_node_ids, self.correlation), dtype=torch.float32).to(self.device)

            # ground truth vector 也放到 GPU
            ground_truth_vector = torch.tensor(ground_truth_vector, dtype=torch.float32).to(self.device)

            # 缓存此节点的数据
            data_cache[node_id] = {
                'ground_truth_vector': ground_truth_vector,
                'm': m,
                'tani': tani,
                'cors': cors,
                'f': f,
                'labels': labels
            }
        
        print(f"所有数据已加载并转移到 {self.device}")
        return data_cache

    def __len__(self):
        return len(self.final)

    def __getitem__(self, idx):
        node_id = self.final.loc[idx, 'ID']
        # 直接从缓存中提取数据
        node_data = self.data_cache[node_id]

        return node_id, (
            node_data['ground_truth_vector'], 
            node_data['m'], 
            node_data['tani'], 
            node_data['cors'], 
            node_data['f'], 
            node_data['labels']
        )


# class ScoringDataset(Dataset):
#     def __init__(self, f_dir, libraryNodes,library_nodes, final, NlibraryNodes, edges, correlation, library_m, tani_library):
#         self.data_dir = f_dir
#         self.libraryNodes = libraryNodes
#         self.library_nodes = library_nodes
#         self.final = final
#         self.non_library_nodes = NlibraryNodes
#         self.edges = edges
#         self.correlation = correlation
#         self.library_m = library_m
#         self.tani_library = tani_library
        
#         # Combine dataframes to create a unified dataset
#         # self.nodes = pd.concat([self.final, self.non_library_nodes], ignore_index=True)
#         self.nodes = final
#         print("Data Initialized")
        
#     def __len__(self):
#         return len(self.nodes)

#     def __getitem__(self, idx):
#         print(f"Fetch data at index {idx}")
#         node_id = self.nodes.loc[idx, 'ID']
#         smiles = self.nodes.loc[idx, 'SMILES']
        
#         # Load corresponding CSV file for this node
#         node_df = loadNodeCsv(node_id, self.data_dir)
#         labels = node_df["SMILES"]
#         ground_truth_index = labels[labels == smiles].index[0]
#         ground_truth_vector = oneHotVect(labels, ground_truth_index)
        
#         # Get connected nodes and relevant attributes
#         connected_node_ids = findConnectedLibrary(node_id, self.edges, self.libraryNodes)
#         connected_smiles = findConnectedSmiles(connected_node_ids, self.libraryNodes)
        
#         # Get the f value
#         f = node_df["Score"]
#         f = torch.tensor(f, dtype=torch.float32)
        
#         # Calculate Tanimoto coefficient
#         tani_list = []
#         for smile in node_df["SMILES"].values:
#             connectedSmiles = findConnectedSmiles(connected_node_ids, self.libraryNodes)
#             # tani = calcTanimotoCoef(smile, connectedSmiles)
#             tani = getTanimotoCoef(smiles, connected_smiles, self.tani_library)
#             tani_list.append(tani)
#         tani = np.array(tani_list)
        
#         # Get m values
#         m = getMVal(connected_node_ids, self.library_m)
        
#         # Get correlation values
#         cors = getCorrelation(node_id, connected_node_ids, self.correlation)
        
#         # Convert everything to PyTorch tensors
#         ground_truth_vector = torch.tensor(ground_truth_vector, dtype=torch.float32)
#         m = torch.tensor(m, dtype=torch.float32)
#         tani = torch.tensor(tani, dtype=torch.float32)
#         cors = torch.tensor(cors, dtype=torch.float32)

#         # giving back node data and other features
#         return node_id, (ground_truth_vector, m, tani, cors, f, labels)

def custom_collate_fn(batch):
    node_ids = [item[0] for item in batch]
    features = [item[1] for item in batch]

    # take out all the features from the zip
    ground_truth_vectors, ms, tanis, corses, fs, labels = zip(*features)

    # take all the features out
    ground_truth_vectors = list(ground_truth_vectors)
    ms = list(ms)
    tanis = list(tanis)
    corses = list(corses)
    fs = list(fs)
    labels = list(labels)

    return node_ids, ground_truth_vectors, ms, tanis, corses, fs, labels


# DataLoader function
def get_dataloader(data_dir, libraryNodes,library_nodes, final, NlibraryNodes, edges, correlation, library_m, tani_library, batch_size, shuffle):
    dataset = ScoringDataset(data_dir, libraryNodes,library_nodes, final, NlibraryNodes, edges, correlation, library_m, tani_library)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)


def load_data(dataset_path, nodeFolder):
    # basic data preprocessing
    # allNodes dataframe
    allNodes = pd.read_csv(dataset_path + "/nodes_data.csv")
    correlation = pd.read_csv(dataset_path + "/cor.csv")
    library_m = pd.read_csv(dataset_path + "/m.csv")
    edges = pd.read_csv(dataset_path + "/edges.csv")
    # NlibraryNodes dataframe
    NlibraryNodes = allNodes[~allNodes["ID"].isin(library_m["ID"])].reset_index(
        drop=True
    )
    libraryNodes = allNodes[allNodes["ID"].isin(library_m["ID"])].reset_index(drop=True)
    libraryNodes = libraryNodes.iloc[::-1]
    
    # library nodes dataframe
    libraryNodes = libraryNodes.reset_index(drop=True)
    # connected library nodes
    connected_library_nodes = edges[
        edges["Library_node"].isin(libraryNodes["ID"])
        & edges["Connected_node"].isin(libraryNodes["ID"])
    ].reset_index(drop=True)

    data = connected_library_nodes

    # renaming
    library_nodes = data[["Library_node", "SMILES"]].rename(
        columns={"Library_node": "ID", "SMILES": "SMILES"}
    )
    connected_nodes = data[["Connected_node", "Expected_SMILES"]].rename(
        columns={"Connected_node": "ID", "Expected_SMILES": "SMILES"}
    )

    # union all
    merged_nodes = pd.concat([library_nodes, connected_nodes]).drop_duplicates()

    missing_files = check_files_exist(merged_nodes, "ID", nodeFolder)

    if len(missing_files) == 0:
        print("all files exist!")
    else:
        print(f"The following files missing: {missing_files}")

    final = merged_nodes[~merged_nodes["ID"].isin(missing_files)].reset_index(drop=True)
    return (
        libraryNodes, 
        library_nodes,
        connected_nodes,
        final,
        NlibraryNodes,
        library_m,
        edges,
        correlation,
    )
    
# Example usage:
if __name__ == "__main__":
    dataset_path = "../datasets/Synthetic"
    f_dir = dataset_path + "/f_filter"
    tani_library = pd.read_csv(dataset_path + "/all_similarity_scores.csv") # TODO: need to complete the file path here
    (
        libraryNodes,
        library_nodes, 
        connected_nodes,
        final,
        NlibraryNodes,
        library_m,
        edges,
        correlation,
    ) = load_data(dataset_path, f_dir)

    # Create DataLoader
    dataloader = get_dataloader(
        f_dir,
        libraryNodes,
        library_nodes,
        final,
        NlibraryNodes,
        edges,
        correlation,
        library_m,
        tani_library,
        batch_size=1,
        shuffle=True
    )
    # Iterate through the DataLoader
    for data in dataloader:
        nodeId, ground_truth_vector, m, tani, cors, f, labels = data
        print(f"Node ID: {nodeId}\n")
        print("============================================================================================\n")
        print(f"Ground Truth Vector: {ground_truth_vector}\n")
        print(len(ground_truth_vector[0]))
        print("============================================================================================\n")
        print(f"f values: {f}\n")
        print(len(f[0]))
        print("============================================================================================\n")
        print(f"m values: {m}\n")
        print(len(m[0]))
        print("============================================================================================\n")
        print(f"Tanimoto Coefficients: {tani}\n")
        print(len(tani[0]))
        print("============================================================================================\n")
        print(f"Correlation values: {cors}\n")
        print(len(cors[0]))
        print("============================================================================================\n")
        print(labels)
        print(len(labels))
        print("============================================================================================\n")
