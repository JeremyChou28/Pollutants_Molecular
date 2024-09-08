import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split

# basic data preprocessing
allNodes = pd.read_csv('nodes_data.csv')
correlation = pd.read_csv('cor.csv')
library_m = pd.read_csv('m.csv')
edges = pd.read_csv('edges.csv')
allNodes = allNodes.drop(['index'], axis=1)
NlibraryNodes = allNodes[~allNodes['ID'].isin(library_m['ID'])].reset_index(drop=True)
libraryNodes = allNodes[allNodes['ID'].isin(library_m['ID'])].reset_index(drop=True)
libraryNodes = libraryNodes.iloc[::-1]
libraryNodes = libraryNodes.reset_index(drop=True)

# load the csv files correspond to the id of the nodes
def loadNodeCsv(nodeId, folderPath):
    filePath = os.path.join(folderPath, f'{nodeId}.csv')
    if os.path.exists(filePath):
        return pd.read_csv(filePath)
    else:   
        raise FileNotFoundError(f"No CSV file found for node ID {nodeId} at {filePath}")
    
def smiles2Fingerprint(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def calcTanimotoCoef(smile, smileSet):
    smile_fp = smiles2Fingerprint(smile)
    tanimoto_scores = []
    for s in smileSet:
        s_fp = smiles2Fingerprint(s)
        tanimoto_scores.append(TanimotoSimilarity(smile_fp, s_fp))
    return tanimoto_scores

def findConnectedLibrary(nodeId, edges, libraryNodes):
    connectedNodes = []
    connectedNodes.extend(edges[edges['Connected_node'] == nodeId]['Library_node'])
    connectedNodes.extend(edges[edges['Library_node'] == nodeId]['Connected_node'])
    connectedDf = pd.Series(connectedNodes)
    connectedLibrary = connectedDf[connectedDf.isin(libraryNodes['ID'])].tolist()
    return connectedLibrary

def findConnectedSmiles(connectedNodes):
    smileSet = []
    smileSet.extend(libraryNodes[libraryNodes['ID'].isin(connectedNodes)]['SMILES'].tolist())
    smileSet.reverse()
    return smileSet

def getCorrelation(nodeId, nodeList):
    result = []
    
    for node in nodeList:
        match = correlation[((correlation['Library_node'] == nodeId) & (correlation['Connected_node']))
                            | ((correlation['Library_node'] == node) & (correlation['Connected_node'] == nodeId))]
    
        if not match.empty:
            corVal = match['Cor'].values[0]
        else:
            corVal = 0
        result.append(corVal)
    
    return result

# functions to get m values
def getMVal(idList, df):
    mVals = []
    for item in idList:
        if item in df['ID'].values:
            mVal = df[df['ID'] == item]['m'].values[0]
        else:
            continue
        mVals.append(mVal)
    
    return mVals

def calcM(nodeId):
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
        
    def forward(self, nodeId, connectedNodeIds):
        node_df = loadNodeCsv(nodeId, self.nodeFolder)
        labels = node_df['SMILES']
        f = torch.tensor(node_df['Score'].values, dtype=torch.float32)
        
        m = calcM(nodeId)
        connectedSmiles = findConnectedSmiles(connectedNodeIds)
        tani = torch.tensor(calcTanimotoCoef(node_df['SMILES'].values[0], connectedSmiles), dtype=torch.float32)
        
        cors = torch.tensor(getCorrelation(nodeId, connectedNodeIds), dtype=torch.float32)
        
        sig = torch.sigmoid(-self.beta * (m * tani - self.gamma))
        s = self.alpha * f + (1-self.alpha) * torch.sum(torch.abs(cors) * sig, dim=0)
        
        return labels, s

def train_and_predict(train_df, test_df, model):
    # define the split ratio here
    train_ratio = 0.8
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1000):
        # prepare data
        train_data, validation_data = train_test_split(train_df, test_size=(1-train_ratio), shuffle=True)
        
        model.train()
        total_loss = 0
        for i, row in train_data.iterrows():
            nodeId = row['ID']
            smiles = row['SMILES']
            connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)
            labels, s = model(nodeId, connected_node_ids)
            
            # do some check
            # print(f"Node ID: {nodeId}, Labels: {labels.tolist()}, Ground Truth: {smiles}")
            
            ground_truth_index = labels[labels == smiles].index[0]
            
            s_probs = nn.functional.softmax(s, dim=0)
            ground_truth_vector = oneHotVect(labels, ground_truth_index)
            
            loss = calcCustomCrossEntropy(s_probs, ground_truth_vector)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Print training loss after each epoch
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}], Loss: {total_loss: .4f}')
        
        # Validate the model
        model.eval()
        validation_correct = 0
        with torch.no_grad():
            for i, row in validation_data.iterrows():
                nodeId = row['ID']
                smiles = row['SMILES']
                connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)
                    
                labels, S_ = model(nodeId, connected_node_ids)
                S_probs = nn.functional.softmax(S_, dim=0)
                predicted_label_index = S_probs.argmax().item()
                predicted_label = labels.iloc[predicted_label_index]
                
                print(f"Node ID: {nodeId}, Labels: {predicted_label}, Ground Truth: {smiles}")
                if predicted_label == smiles:
                    validation_correct += 1
            
        validation_acc = validation_correct / len(validation_data)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}], Validation Acc: {validation_acc: .4f}')
    
    # Testing on test data
    print("Testing on test data...")
    test_correct = 0
    with torch.no_grad():
        for i, row in test_df.iterrows():
            nodeId = row['ID']
            smiles = row['SMILES']
            connected_node_ids = findConnectedLibrary(nodeId, edges, libraryNodes)
                
            labels, s__ = model(nodeId, connected_node_ids)
            s__probs = nn.functional.softmax(s__, dim=0)
            predicted_label_index = s__probs.argmax().item()
            predicted_label = labels.iloc[predicted_label_index]
            
            if predicted_label == smiles:
                test_correct += 1
        
    test_acc = test_correct / len(test_df)
    print(f'Test Acc: {test_acc: .4f}')
    print(f'Final alpha: {model.alpha.item(): .4f}')
    print(f"Final beta: {model.beta.item(): .4f}")
    print(f"Final gamma: {model.gamma.item(): .4f}")

# Load data and start training
train_df = libraryNodes
test_df = NlibraryNodes

# Initialize the model
nodeFolder = "./f"
connected_node_folder = "./outputs"
model = ScoringModel(nodeFolder, connected_node_folder)

# Start training and prediction
train_and_predict(train_df, test_df, model)

