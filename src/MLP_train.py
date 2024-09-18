import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import train_test_split

# rebuild the code later

# basic data preprocessing
allNodes = pd.read_csv('nodes_data.csv')
correlation = pd.read_csv('cor.csv')
library_m = pd.read_csv('m.csv')
edges = pd.read_csv('edges.csv')
NlibraryNodes = allNodes[~allNodes['ID'].isin(library_m['ID'])].reset_index(drop=True)
libraryNodes = allNodes[allNodes['ID'].isin(library_m['ID'])].reset_index(drop=True)
libraryNodes = libraryNodes.iloc[::-1]
libraryNodes = libraryNodes.reset_index(drop=True)
# connected library nodes
connected_library_nodes = edges[edges['Library_node'].isin(libraryNodes['ID']) & edges['Connected_node'].isin(libraryNodes['ID'])].reset_index(drop=True)

data = connected_library_nodes

# renaming
library_nodes = data[['Library_node', 'SMILES']].rename(columns={'Library_node': 'ID', 'SMILES': 'SMILES'})
connected_nodes = data[['Connected_node', 'Expected_SMILES']].rename(columns={'Connected_node': 'ID', 'Expected_SMILES': 'SMILES'})

# union all
merged_nodes = pd.concat([library_nodes, connected_nodes]).drop_duplicates()

folder_path = './f_filter'

# check the existence of the files in f_filter
def check_files_exist(df, column_name, folder_path):
    missing_files = []
    for item in df[column_name]:
        file_path = os.path.join(folder_path, f"{item}.csv")
        if not os.path.isfile(file_path):
            missing_files.append(item)
    return missing_files

missing_files = check_files_exist(merged_nodes, 'ID', folder_path)

if len(missing_files) == 0:
    print("所有文件都存在")
else:
    print(f"以下文件缺失: {missing_files}")

final = merged_nodes[~merged_nodes['ID'].isin(missing_files)].reset_index(drop=True)



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

    return mVals


def calcM(nodeId, edges, libraryNodes, library_m):
    connected_lib = findConnectedLibrary(nodeId, edges, libraryNodes)
    m = getMVal(connected_lib, library_m)
    return torch.tensor(m, dtype=torch.float32)


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


# define the model
class ScoringModel(nn.Module):
    def __init__(self, nodeFolder, connectedNodeFolder):
        super(ScoringModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.3))
        self.beta = nn.Parameter(torch.tensor(-9.0))
        self.gamma = nn.Parameter(torch.tensor(0.6))

        self.nodeFolder = nodeFolder
        self.connectedNodeFolder = connectedNodeFolder
    
    def forward(
        self, nodeId, connectedNodeIds, edges, libraryNodes, library_m, correlation
    ):
        node_df = loadNodeCsv(nodeId, self.nodeFolder)
        labels = node_df["SMILES"]
        f = torch.tensor(node_df["Score"].values, dtype=torch.float32)

        # get tanimoto for each of the connected nodes
        tani_list = []
        for smiles in node_df["SMILES"].values:
            connectedSmiles = findConnectedSmiles(connectedNodeIds, libraryNodes)
            tani = calcTanimotoCoef(smiles, connectedSmiles)
            tani_list.append(tani)

        tani = torch.tensor(tani_list, dtype=torch.float32)

        m = calcM(nodeId, edges, libraryNodes, library_m)
    
        # print(f"tani: {tani}")
        # print(f"m: {m}")
        # print(f"m * tani: {m * tani}")
    
        cors = torch.tensor(
            getCorrelation(nodeId, connectedNodeIds, correlation), dtype=torch.float32
        )
        sig = torch.sigmoid(-self.beta * (m * tani - self.gamma))
        # s = self.alpha * f + (1 - self.alpha) * torch.sum(torch.abs(cors) * sig, dim=1)
        s = self.alpha * f + (1 - self.alpha) * torch.sum(sig)

        return labels, s




def train_and_predict(
    train_df, test_df, model, edges, libraryNodes, library_m, correlation
):
    # define the split ratio here
    train_ratio = 0.8
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        # prepare data
        train_data, validation_data = train_test_split(
            train_df, test_size=(1 - train_ratio), shuffle=True
        )

        model.train()
        total_loss = 0
        for i, row in train_data.iterrows():
            nodeId = row["ID"]
            smiles = row["SMILES"]
            connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)
            labels, s = model(
                nodeId, connected_node_ids, edges, libraryNodes, library_m, correlation
            )

            # do some check
            # print(f"Node ID: {nodeId}, Labels: {labels.tolist()}, Ground Truth: {smiles}")

            ground_truth_index = labels[labels == smiles].index[0]

            s_probs = F.softmax(s, dim=0)
            # if count < 100000:
            #     print(f"s: {s}")
            #     print(f"s_probs: {s_probs}")
            #     print(f"alpha: {model.alpha.item()}")
            #     count = count + 1
            # else:
            #     count = count
            ground_truth_vector = oneHotVect(labels, ground_truth_index)

            loss = calcCustomCrossEntropy(s_probs, ground_truth_vector)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training loss after each epoch
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}], Loss: {total_loss: .4f}\n")

        # Validate the model
        model.eval()
        validation_correct = 0
        with torch.no_grad():
            for i, row in validation_data.iterrows():
                nodeId = row["ID"]
                smiles = row["SMILES"]
                connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)

                labels, S_ = model(
                    nodeId,
                    connected_node_ids,
                    edges,
                    libraryNodes,
                    library_m,
                    correlation,
                )
                S_probs = F.softmax(S_, dim=0)
                predicted_label_index = S_probs.argmax().item()
                predicted_label = labels.iloc[predicted_label_index]

                # print(
                    # f"Node ID: {nodeId}, Labels: {predicted_label}, Ground Truth: {smiles}"
                # )
                if predicted_label == smiles:
                    validation_correct += 1

        validation_acc = validation_correct / len(validation_data)
        # if (epoch + 1) % 10 == 0: 
        print(f"Epoch [{epoch + 1}], Validation Acc: {validation_acc: .4f}\n")

    # Testing on test data
    print("Testing on test data...")
    test_correct = 0
    with torch.no_grad():
        for i, row in test_df.iterrows():
            nodeId = row["ID"]
            smiles = row["SMILES"]
            connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)

            labels, s__ = model(
                nodeId,
                connected_node_ids,
                edges,
                libraryNodes,
                library_m,
                correlation,
            )
            s__probs = F.softmax(s__, dim=0)
            predicted_label_index = s__probs.argmax().item()
            predicted_label = labels.iloc[predicted_label_index]
            
            # write in the file
            with open("./test_exact.txt", "a") as f:
                f.write(f"node ID: {nodeId}, predicted SMILE:{predicted_label}, ground truth SMILE: {smiles}")
                f.write("\n")
                f.close
            
            if predicted_label == smiles:
                test_correct += 1

    test_acc = test_correct / len(test_df)
    print(f"Test Acc: {test_acc: .4f}")
    print(f"Final alpha: {model.alpha.item(): .4f}")
    print(f"Final beta: {model.beta.item(): .4f}")
    print(f"Final gamma: {model.gamma.item(): .4f}")
    
    with open("./test_exact.txt", "a") as f:
        f.write(f"Test Acc: {test_acc: .4f} \n")
        f.write(f"Final alpha: {model.alpha.item(): .4f} \n")
        f.write(f"Final beta: {model.beta.item(): .4f} \n")
        f.write(f"Final gamma: {model.gamma.item(): .4f} \n")
        f.write("\n")
        f.close


def main(args):
    # basic data preprocessing
    allNodes = pd.read_csv(args.dataset_path + "/nodes_data.csv")
    correlation = pd.read_csv(args.dataset_path + "/cor.csv")
    library_m = pd.read_csv(args.dataset_path + "/m.csv")
    edges = pd.read_csv(args.dataset_path + "/edges.csv")
    NlibraryNodes = allNodes[~allNodes["ID"].isin(library_m["ID"])].reset_index(
        drop=True
    )
    libraryNodes = allNodes[allNodes["ID"].isin(library_m["ID"])].reset_index(drop=True)
    libraryNodes = libraryNodes.iloc[::-1]
    libraryNodes = libraryNodes.reset_index(drop=True)

    # Load data and start training
    train_df = final
    test_df = NlibraryNodes

    # Initialize the model
    nodeFolder = args.dataset_path + "/f_filter"
    connected_node_folder = args.dataset_path + "/outputs"
    model = ScoringModel(nodeFolder, connected_node_folder)

    # Start training and prediction
    train_and_predict(
        train_df, test_df, model, edges, libraryNodes, library_m, correlation
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../Sediment_check",
        help="The folder path of dataset",
    )
    args = parser.parse_args()

    main(args)
