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
from models.MLP import *
from utils.utils import *
from dataloader.dataloader import *


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


def main(args):
    # load args
    dataset_path = args.dataset_path
    dataset = dataset_path.split("/")[-1]
    result_path = args.result_path
    checkpoint_path = args.checkpoint_path
    nodeFolder = dataset_path + "/f_filter"
    train_ratio = args.train_sample_ratio
    learning_rate = args.learning_rate
    epochs = args.epochs
    peak_pick_num = args.peak_pick_num
    method_variant = args.method_variant
    tani_library = pd.read_csv(dataset_path + "/all_similarity_scores.csv")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # set seed
    seed_torch(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load data
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
    print(f"train_df len: {len(train_df)}")
    test_df = NlibraryNodes
    print(f"test_df len: {len(test_df)}")
    total_dataframe = pd.concat([train_df, test_df], ignore_index=True)
    
    batch_size = 1

    # don't define the validation set yet, due to very little amount of data
    train_loader = get_dataloader(nodeFolder, libraryNodes, library_nodes, train_df, NlibraryNodes, edges, correlation, library_m, tani_library, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(nodeFolder, libraryNodes, library_nodes, test_df, NlibraryNodes, edges, correlation, library_m, tani_library, batch_size=batch_size, shuffle=False)
    
    print(f"train_loader len: {len(train_loader)}")
    print(f"test_loader len: {len(test_loader)}")
    print("=========================================Successfully load the data!====================================================================================")
    
    # Define model and optimizer
    if method_variant == "wo_cor":
        model = Metfusion().to(device)
    elif method_variant == "cor_inside":
        model = Metfusion_Cor_Inside().to(device)
    elif method_variant == "cor_outside":
        model = Metfusion_Cor_Outside().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_acc = 0
    best_val_acc = 0
    if args.scratch:
        # Training
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_idx, data in enumerate(train_loader):
                print(f"Loading batch {batch_idx}")
                nodeId, ground_truth_vector, m, tani, cors, f, _ = data
                nodeId = nodeId[0]
                smiles = total_dataframe[total_dataframe["ID"] == nodeId]["SMILES"].values[0]
                m = (m[0]).to(device)
                cors = (cors[0]).to(device)
                tani = (tani[0]).to(device)
                f = (f[0]).to(device)
                ground_truth_vector = (ground_truth_vector[0]).to(device)
                
                s = model(m, f, tani, cors)
                s_probs = F.softmax(s, dim=0)
                loss = calcCustomCrossEntropy(s_probs, ground_truth_vector)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                

            print(f"Epoch [{epoch + 1}], Loss: {total_loss/len(train_loader): .4f}\n")

            model.eval()
            test_correct = 0
            with torch.no_grad():
                for data in test_loader:
                    nodeId, ground_truth_vector, m, tani, cors, f, labels = data
                    nodeId = nodeId[0]
                    smiles = total_dataframe[total_dataframe["ID"] == nodeId]["SMILES"].values[0]
                    m = (m[0]).to(device)
                    cors = (cors[0]).to(device)
                    tani = (tani[0]).to(device)
                    f = (f[0]).to(device)
                    ground_truth_vector = (ground_truth_vector[0]).to(device)
                    
                    s = model(m, f, tani, cors)
                    s_probs = F.softmax(s, dim=0)
                    top_n_probs, top_n_indices = torch.topk(s_probs, peak_pick_num)
                    top_n_indices = top_n_indices.cpu().numpy()
                    top_n_smiles = (labels[0]).iloc[top_n_indices].tolist()
                    if smiles in top_n_smiles:
                        test_correct += 1
                    else:
                        continue
                
            test_acc = test_correct / len(test_df)
            print(f"Epoch [{epoch + 1}], Test Acc: {test_acc: .4f}\n")
        
            # saving the best model on validation set
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        checkpoint_path,
                        f"{dataset}_{method_variant}_top{peak_pick_num}_best_model.pth",
                    ),
                )
                print("=============================================================Successfully saved the checkpoint========================================================================================")

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
        for data in test_loader:
            nodeId, ground_truth_vector, m, tani, cors, f, labels = data
            nodeId = nodeId[0]
            smiles = test_df[test_df["ID"] == nodeId]["SMILES"].values[0]
            m = (m[0]).to(device)
            cors = (cors[0]).to(device)
            tani = (tani[0]).to(device)
            f = (f[0]).to(device)
            ground_truth_vector = (ground_truth_vector[0]).to(device)
            
            s = model(m, f, tani, cors)
            s_probs = F.softmax(s, dim=0)
            top_n_probs, top_n_indices = torch.topk(s_probs, peak_pick_num)
            top_n_indices = top_n_indices.cpu().numpy()
            top_n_smiles = (labels[0]).iloc[top_n_indices].tolist()
            if smiles in top_n_smiles:
                test_correct += 1
            else:
                continue
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
    
    # write in the file
    with open(
        result_path + f"/{dataset}_{method_variant}_top{peak_pick_num}.txt",
        "w+",
    ) as f:
        for result in result_list:
            f.write(result)
        f.close()
    test_acc = test_correct / len(test_df)
    print(f"Test Acc: {test_acc: .4f}")
    print(f"Final alpha: {model.alpha.item(): .4f}")
    print(f"Final beta: {model.beta.item(): .4f}")
    print(f"Final gamma: {model.gamma.item(): .4f}")

def str2bool(string):
    if string.lower() in ["true", "t", "yes", "y"]:
        return True
    elif string.lower() in ["false", "f", "no", "n"]:
        return False
    else:
        raise ValueError("Invalid boolean value")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pollutants molecular network prediction."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../datasets/Sediment_check",
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
        default="wo_cor",
        choices=["wo_cor", "cor_inside", "cor_outside"],
    )
    parser.add_argument(
        "--peak_pick_num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--scratch", type=str2bool, default="True", help="Train from scratch"
    )
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    start_time = time.time()
    main(args)
    print("Spend time: {}".format(time.time() - start_time))