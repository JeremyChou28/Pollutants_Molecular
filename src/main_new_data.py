import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import train_test_split

sys.path.append("../")
from models.model import *
from utils.utils import *


# check the existence of the files in f_filter
def check_files_exist(df, column_name, folder_path):
    missing_files = []
    for item in df[column_name]:
        file_path = os.path.join(folder_path, f"{item}.csv")
        if not os.path.isfile(file_path):
            missing_files.append(item)
    return missing_files


# load the csv files correspond to the id of the nodes
def loadNodeCsv(nodeId, folderPath):
    filePath = os.path.join(folderPath, f"{nodeId}.csv")
    if os.path.exists(filePath):
        return pd.read_csv(filePath)
    else:
        raise FileNotFoundError(f"No CSV file found for node ID {nodeId} at {filePath}")


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


def findConnectedLibrary(nodeId, edges, libraryNodes):
    connectedNodes = []
    connectedNodes.extend(edges[edges["Connected_node"] == nodeId]["Library_node"])
    connectedNodes.extend(edges[edges["Library_node"] == nodeId]["Connected_node"])
    connectedDf = pd.Series(connectedNodes)
    connectedLibrary = connectedDf[connectedDf.isin(libraryNodes["ID"])].tolist()
    return connectedLibrary


def findConnectedSmiles(connectedNodes, libraryNodes):
    smileSet = []
    smileSet.extend(
        libraryNodes[libraryNodes["ID"].isin(connectedNodes)]["SMILES"].tolist()
    )
    smileSet.reverse()
    return smileSet


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


# functions to get m values
def getMVal(idList, df):
    mVals = []
    for item in idList:
        if item in df["ID"].values:
            mVal = df[df["ID"] == item]["m"].values[0]
        else:
            continue
        mVals.append(mVal)

    return np.array(mVals)


# one hot vector function for cross entropy
def oneHotVect(labels, index):
    vector = torch.zeros(len(labels), dtype=torch.float32)
    vector[index] = 1.0
    return vector


# custom cross entropy function
def calcCustomCrossEntropy(predicted_probs, ground_truth_vector):
    log_likelihood = -torch.log(predicted_probs)
    crossEntropy = torch.sum(ground_truth_vector * log_likelihood)
    return crossEntropy


def load_data(dataset_path, test_subset_index):
    (
        test_libraryNodes,
        test_NlibraryNodes,
        test_library_m,
        test_edges,
        test_correlation,
    ) = load_sub_data(dataset_path + f"/subset_{test_subset_index}")

    (
        train_libraryNodes_dict,
        train_NlibraryNodes_dict,
        train_library_m_dict,
        train_edges_dict,
        train_correlation_dict,
    ) = ({}, {}, {}, {}, {})
    train_subset_index_list = [i for i in range(1, 6) if i != test_subset_index]
    for i in train_subset_index_list:
        subset_path = dataset_path + f"/subset_{i}"
        (
            libraryNodes,
            NlibraryNodes,
            library_m,
            edges,
            correlation,
        ) = load_sub_data(subset_path)
        train_libraryNodes_dict[f"subset_{i}"] = libraryNodes
        train_NlibraryNodes_dict[f"subset_{i}"] = NlibraryNodes
        train_library_m_dict[f"subset_{i}"] = library_m
        train_edges_dict[f"subset_{i}"] = edges
        train_correlation_dict[f"subset_{i}"] = correlation

    train_data = [
        train_libraryNodes_dict,
        train_NlibraryNodes_dict,
        train_library_m_dict,
        train_edges_dict,
        train_correlation_dict,
    ]
    test_data = [
        test_libraryNodes,
        test_NlibraryNodes,
        test_library_m,
        test_edges,
        test_correlation,
    ]
    return train_data, test_data


def load_sub_data(dataset_path):
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
    return (
        libraryNodes,
        NlibraryNodes,
        library_m,
        edges,
        correlation,
    )


def data_preprocessing(
    nodeId,
    smiles,
    edges,
    libraryNodes,
    library_m,
    correlation,
    nodeFolder,
    tani_lib_folder,
):
    connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)

    node_df = loadNodeCsv(nodeId, nodeFolder)
    labels = node_df["SMILES"]
    ground_truth_index = labels[labels == smiles].index[0]
    ground_truth_vector = oneHotVect(labels, ground_truth_index)

    tani = np.load(tani_lib_folder + f"/{nodeId}.npy")

    connected_lib = findConnectedLibrary(nodeId, edges, libraryNodes)
    m = getMVal(connected_lib, library_m)

    cors = getCorrelation(nodeId, connected_node_ids, correlation)
    return node_df, tani, m, cors, ground_truth_vector, labels


def main(args):
    # load args
    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[-1]
    result_path = args.result_path
    checkpoint_path = args.checkpoint_path

    train_ratio = args.train_sample_ratio
    learning_rate = args.learning_rate
    epochs = args.epochs
    peak_pick_num = args.peak_pick_num
    method_variant = args.method_variant

    # set seed
    seed_torch(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load data
    train_dataset, test_dataset = load_data(dataset_path, args.subset_index)
    (
        train_libraryNodes_dict,
        train_NlibraryNodes_dict,
        train_library_m_dict,
        train_edges_dict,
        train_correlation_dict,
    ) = train_dataset
    (
        test_libraryNodes,
        test_NlibraryNodes,
        test_library_m,
        test_edges,
        test_correlation,
    ) = test_dataset

    train_subset_index_list = [i for i in range(1, 6) if i != args.subset_index]

    test_nodeFolder = dataset_path + f"/subset_{args.subset_index}/f"
    test_tani_lib_folder = dataset_path + f"/subset_{args.subset_index}/train_tani_lib"

    # Define model and optimizer
    if method_variant == "wo_cor":
        model = Metfusion().to(device)
    elif method_variant == "cor_inside":
        model = Metfusion_Cor_Inside().to(device)
    elif method_variant == "cor_outside":
        model = Metfusion_Cor_Outside().to(device)
    elif method_variant == "cor_our":
        model = Metfusion_Our().to(device)
    elif method_variant == "cor_our_v2":
        model = Metfusion_Our_v2().to(device)
    elif method_variant == "cor_our_v3":
        model = Metfusion_Our_v3(hidden_size=64).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_trial_test_acc = 0
    best_val_acc = 0
    if args.scratch:
        # Training
        for epoch in range(epochs):

            for index in train_subset_index_list:
                train_libraryNodes = train_libraryNodes_dict[f"subset_{index}"]
                train_NlibraryNodes = train_NlibraryNodes_dict[f"subset_{index}"]
                train_library_m = train_library_m_dict[f"subset_{index}"]
                train_edges = train_edges_dict[f"subset_{index}"]
                train_correlation = train_correlation_dict[f"subset_{index}"]

                train_nodeFolder = dataset_path + f"/subset_{index}/f"
                train_tani_lib_folder = dataset_path + f"/subset_{index}/train_tani_lib"

                # prepare data
                train_data, validation_data = train_test_split(
                    train_NlibraryNodes, test_size=(1 - train_ratio), shuffle=True
                )

                model.train()
                total_loss = 0
                for i, row in train_data.iterrows():
                    nodeId = row["ID"]
                    smiles = row["SMILES"]

                    node_df, tani, m, cors, ground_truth_vector, labels = (
                        data_preprocessing(
                            nodeId,
                            smiles,
                            train_edges,
                            train_libraryNodes,
                            train_library_m,
                            train_correlation,
                            train_nodeFolder,
                            train_tani_lib_folder,
                        )
                    )

                    f = torch.tensor(node_df["Score"].values, dtype=torch.float32).to(
                        device
                    )
                    tani = torch.tensor(tani, dtype=torch.float32).to(device)
                    m = torch.tensor(m, dtype=torch.float32).to(device)
                    cors = torch.tensor(cors, dtype=torch.float32).to(device)
                    ground_truth_vector = ground_truth_vector.to(device)

                    s = model(m, f, tani, cors)
                    s_probs = F.softmax(s, dim=0)

                    loss = calcCustomCrossEntropy(s_probs, ground_truth_vector)
                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f"Epoch [{epoch + 1}], Loss: {total_loss/len(train_data): .4f}\n")

                # Validate the model
                model.eval()
                validation_correct = 0
                with torch.no_grad():
                    for i, row in validation_data.iterrows():
                        nodeId = row["ID"]
                        smiles = row["SMILES"]
                        node_df, tani, m, cors, ground_truth_vector, labels = (
                            data_preprocessing(
                                nodeId,
                                smiles,
                                train_edges,
                                train_libraryNodes,
                                train_library_m,
                                train_correlation,
                                train_nodeFolder,
                                train_tani_lib_folder,
                            )
                        )

                        f = torch.tensor(
                            node_df["Score"].values, dtype=torch.float32
                        ).to(device)
                        tani = torch.tensor(tani, dtype=torch.float32).to(device)
                        m = torch.tensor(m, dtype=torch.float32).to(device)
                        cors = torch.tensor(cors, dtype=torch.float32).to(device)
                        ground_truth_vector = ground_truth_vector.to(device)

                        s = model(m, f, tani, cors)
                        s_probs = F.softmax(s, dim=0)

                        predicted_label_index = s_probs.argmax().item()
                        predicted_label = labels.iloc[predicted_label_index]

                        # print(
                        # f"Node ID: {nodeId}, Labels: {predicted_label}, Ground Truth: {smiles}"
                        # )
                        if predicted_label == smiles:
                            validation_correct += 1

                validation_acc = validation_correct / len(validation_data)
                print(f"Epoch [{epoch + 1}], Validation Acc: {validation_acc: .4f}\n")

                # saving the best model on validation set
                if validation_acc >= best_val_acc:
                    best_val_acc = validation_acc
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            checkpoint_path,
                            f"{dataset}_{method_variant}_top{peak_pick_num}_best_model.pth",
                        ),
                    )

    # Testing
    model.load_state_dict(
        torch.load(
            os.path.join(
                checkpoint_path,
                f"{dataset}_{method_variant}_top{peak_pick_num}_best_model.pth",
            )
        )
    )
    print("Testing on test data...")
    test_correct = 0
    result_list = []
    with torch.no_grad():
        for i, row in test_NlibraryNodes.iterrows():
            nodeId = row["ID"]
            smiles = row["SMILES"]

            node_df, tani, m, cors, ground_truth_vector, labels = data_preprocessing(
                nodeId,
                smiles,
                test_edges,
                test_libraryNodes,
                test_library_m,
                test_correlation,
                test_nodeFolder,
                test_tani_lib_folder,
            )

            f = torch.tensor(node_df["Score"].values, dtype=torch.float32).to(device)
            tani = torch.tensor(tani, dtype=torch.float32).to(device)
            m = torch.tensor(m, dtype=torch.float32).to(device)
            cors = torch.tensor(cors, dtype=torch.float32).to(device)
            ground_truth_vector = ground_truth_vector.to(device)

            s = model(m, f, tani, cors)
            s_probs = F.softmax(s, dim=0)
            """
            # choose whether to use different choosing strategy and uncomment the one you want to use
            
            """

            top_n_probs, top_n_indices = torch.topk(s_probs, peak_pick_num)
            top_n_indices = top_n_indices.cpu().numpy()
            top_n_smiles = labels.iloc[top_n_indices].tolist()

            result = (
                f"node ID: {nodeId}, ground truth SMILE: {smiles}"
                + "\n"
                + f"top {peak_pick_num} test cadidates: \n"
            )
            candidate = ""
            for idx, value, prob in zip(
                top_n_indices,
                top_n_smiles,
                top_n_probs,
            ):
                candidate += f"smiles: {value}, s: {prob.item(): .5f}" + "\n"
            result += candidate
            result_list.append(result)

            if smiles in top_n_smiles:
                test_correct += 1
            else:
                continue
            """
            predicted_label_index = s_probs.argmax().item()
            predicted_label = labels.iloc[predicted_label_index]
            # write in the file
            with open(result_path + f"/{args.dataset}_test_exact.txt", "a") as f:
                f.write(
                    f"node ID: {nodeId}, predicted SMILE:{predicted_label}, ground truth SMILE: {smiles}"
                )
                f.write("\n")
                f.close
            if predicted_label == smiles:
                test_correct += 1
            """
            # write in the file
    with open(
        result_path
        + f"/{dataset}_{args.subset_index}_{method_variant}_top{peak_pick_num}.txt",
        "w+",
    ) as f:
        for result in result_list:
            f.write(result)
        f.close()
    test_acc = test_correct / len(test_NlibraryNodes)
    print(f"Test Acc: {test_acc: .4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pollutants molecular network prediction."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../datasets/new_data/",
        help="The folder path of dataset",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="../results",
        help="The folder path of saving results",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../checkpoints",
        help="The folder path of saving model checkpoints",
    )
    parser.add_argument(
        "--train_sample_ratio",
        type=float,
        default=0.8,
        help="The ratio of sampled nodes in library nodes during training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--method_variant",
        type=str,
        default="cor_our_v3",
        choices=[
            "wo_cor",
            "cor_inside",
            "cor_outside",
            "cor_our",
            "cor_our_v2",
            "cor_our_v3",
        ],
    )
    parser.add_argument(
        "--peak_pick_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--scratch", type=str2bool, default="True", help="Train from scratch"
    )
    parser.add_argument(
        "--subset_index", type=int, default=1, help="The index of the subset"
    )
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    start_time = time.time()
    main(args)
    print("Spend time: {}".format(time.time() - start_time))
