import os
import time
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
RDLogger.DisableLog("rdApp.*")


# check the existence of the files in f_filter
def check_files_exist(df, column_name, folder_path):
    missing_files = []
    for item in df[column_name]:
        file_path = os.path.join(folder_path, f"{item}.csv")
        if not os.path.isfile(file_path):
            missing_files.append(item)
    return missing_files


def load_data(dataset_path, nodeFolder):
    # basic data preprocessing
    allNodes = pd.read_csv(dataset_path + "/nodes_data.csv")
    correlation = pd.read_csv(dataset_path + "/cor.csv")
    library_m = pd.read_csv(dataset_path + "/m.csv")
    edges = pd.read_csv(dataset_path + "/edges.csv")
    NlibraryNodes = allNodes[~allNodes["ID"].isin(library_m["ID"])].reset_index(
        drop=True
    )
    libraryNodes = allNodes[allNodes["ID"].isin(library_m["ID"])].reset_index(drop=True)
    libraryNodes = libraryNodes.iloc[::-1]
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


def findConnectedLibrary(nodeId, edges, libraryNodes):
    connectedNodes = []
    connectedNodes.extend(edges[edges["Connected_node"] == nodeId]["Library_node"])
    connectedNodes.extend(edges[edges["Library_node"] == nodeId]["Connected_node"])
    connectedDf = pd.Series(connectedNodes)
    connectedLibrary = connectedDf[connectedDf.isin(libraryNodes["ID"])].tolist()
    return connectedLibrary


# load the csv files correspond to the id of the nodes
def loadNodeCsv(nodeId, folderPath):
    filePath = os.path.join(folderPath, f"{nodeId}.csv")
    if os.path.exists(filePath):
        return pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"No CSV file found for node ID {nodeId} at {filePath}")


def findConnectedSmiles(connectedNodes, libraryNodes):
    smileSet = []
    smileSet.extend(
        libraryNodes[libraryNodes["ID"].isin(connectedNodes)]["SMILES"].tolist()
    )
    smileSet.reverse()
    return smileSet


def smiles2Fingerprint(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        # raise ValueError(f"Invalid SMILES string: {smile}")
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


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


def data_preprocessing(edges, libraryNodes, nodeFolder, dataframe, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for i, row in dataframe.iterrows():
        nodeId = row["ID"]
        connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)
        node_df = loadNodeCsv(nodeId, nodeFolder)

        # find the max len to construct matrix
        max_len = max(
            [
                len(findConnectedSmiles(connected_node_ids, libraryNodes))
                for _, row_ in node_df.iterrows()
            ]
        )

        scores = np.zeros((len(node_df), max_len))

        # fillin
        for i_, row_ in node_df.iterrows():
            connectedSmiles = findConnectedSmiles(connected_node_ids, libraryNodes)
            tani = calcTanimotoCoef(row_["SMILES"], connectedSmiles)
            scores[i_, : len(tani)] = tani

        np.save(f"{result_path}/{nodeId}.npy", scores)
        print(f"Node {nodeId} done!")

    return 0


def main(args):
    dataset_path = args.dataset_path
    result_path = args.result_path
    nodeFolder = dataset_path + "/f"
    (
        libraryNodes,
        library_nodes,
        connected_nodes,
        final,
        NlibraryNodes,
        library_m,
        edges,
        correlation,
    ) = load_data(dataset_path, nodeFolder)
    train_df = final
    test_df = NlibraryNodes
    data_preprocessing(edges, libraryNodes, nodeFolder, train_df, result_path)
    data_preprocessing(edges, libraryNodes, nodeFolder, test_df, result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tanimoto Coefficient Library Construction."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../datasets/ChangjiangRiver_check",
        help="The folder path of the datasets, mainly to decide which datasets you want to build on",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../datasets/ChangjiangRiver_check/train_tani_lib",
        help="basically should be consistent with what you set in dataset_path argument",
    )
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    start_time = time.time()
    main(args)
    print("Spend time: {}".format(time.time() - start_time))
