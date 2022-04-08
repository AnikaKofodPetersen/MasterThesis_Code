##################################################
#################### Intro #######################
# This script aims at training my model using cross fold validation
# Author: Anika Kofod Petersen
##################################################

#Import stuff
import os
import re
import sys
import time
import math
import warnings
import pandas as pd
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sn
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import GroupKFold
import pickle

######################
#### Define Paths ####
######################

logfile_PATH = "/home/projects/ht3_aim/people/anipet/new_model_training/logfile2.txt"
models_PATH = "/home/projects/ht3_aim/people/anipet/new_model_training/models2/"

with open(logfile_PATH, 'a+') as f:
        f.write(f'####  Ready to begin  ####\n') 

#Set device to cpu or gpu
device = 'cpu'

#Assign embedding folder
ESM_EMB_PATH = "/home/projects/ht3_aim/people/anipet/model_training/ESM_embeddings/"
PS_EMB_PATH = "/home/projects/ht3_aim/people/anipet/model_training/PS_embeddings/"
CleanedData_PATH = "/home/projects/ht3_aim/people/anipet/model_training/CleanedData.csv"
Cluster_PATH = "/home/projects/ht3_aim/people/anipet/model_training/DB_clu_50_id.tsv"

###########################################
#### Load labels from csv and clean up ####
###########################################
#NESG Normalization function
def NESGNormalizeData(data):
    if data != 6:
        return (data - 0) / (5 - 0)
    else:
        return 6
        
#Load the data
org_df = pd.read_csv(CleanedData_PATH , sep=",")


#Get list of emb IDs
emb_id = list(os.listdir(ESM_EMB_PATH))
emb_id = [idx.split(".")[0].split("_")[-1] for idx in emb_id]


#Drop rows without sequence embeddings (should be none)
df = org_df[org_df.ID.isin(emb_id)]
df = df.reset_index(drop=True)

#Replace NaN with 9 for later ignore index
df.NESG_label.fillna(6, inplace=True)
df.PSI_BIO_label.fillna(9, inplace=True)

#Make sure it is integers
df["NESG_label"] = df["NESG_label"].astype(int)
df["PSI_BIO_label"] = df["PSI_BIO_label"].astype(int)

#Reverse the Psi-Bio label
df["PSI_BIO_label"] = [2 if x==0 else x for x in df["PSI_BIO_label"]]
df["PSI_BIO_label"] = [0 if x==1 else x for x in df["PSI_BIO_label"]]
df["PSI_BIO_label"] = [1 if x==2 else x for x in df["PSI_BIO_label"]]

#Load labels 
NESG_label = list(df["NESG_label"])
psi_bio_label = list(df["PSI_BIO_label"])

#Normalize nesg label between 0 and 1
norm_NESG_label = [ NESGNormalizeData(x) for x in NESG_label]

df["norm_NESG_label"] = norm_NESG_label

#Load cluster annotation
clusters = pd.read_csv(Cluster_PATH, sep="\t",  header=None)
clusters= clusters.rename(columns={0: 'rep', 1 :'id'})

#Make a cluster dictionary
cluster_temp_dict = {}
cluster_dict = {}
count = 0
for i, row in clusters.iterrows():
    if row["rep"] in cluster_temp_dict:
        cluster_dict[row["id"]] = cluster_temp_dict[row["rep"]]
    else:
        cluster_temp_dict[row["rep"]] = count 
        count += 1
        cluster_dict[row["id"]] = cluster_temp_dict[row["rep"]]
        
#Append cluster info to df
clusters = []
for i, row in df.iterrows():
    name = row["ID"]
    clusters.append(cluster_dict[name])
df["cluster"] = clusters        
        
#Update log        
with open(logfile_PATH, 'a+') as f:
        f.write(f'Ready to load embeddings\n')        

#############################
#### Load ESM embeddings ####
#############################


#Load and format embeddings in a dict
ESM_embs_dict = dict()     
for file in os.listdir(ESM_EMB_PATH):
    name = file.split(".")[0].split("_")[-1]
    if file.endswith(".pt") and name in list(df["ID"]):
        tensor_in = torch.load(f'{ESM_EMB_PATH}/{file}')
        ESM_embs_dict[name] = tensor_in
        
#Sanity check
assert len(ESM_embs_dict) == len(os.listdir(ESM_EMB_PATH))

#Update log        
with open(logfile_PATH, 'a+') as f:
        f.write(f'Loaded ESM embeddings\n')  
 
 
#######################################
#### Load ProteinSolver embeddings ####
#######################################

#Load and format embeddings
PS_embs_dict = dict()
for file in os.listdir(PS_EMB_PATH):
    name = file.split(".")[0].split("_")[-1]
    if file.endswith(".pt") and name in list(df["ID"]):
        tensor_in = torch.load(f'{PS_EMB_PATH}/{file}')
        PS_embs_dict[name] = tensor_in
        
#Sanity check
assert len(PS_embs_dict) == len(os.listdir(PS_EMB_PATH))

#Update log        
with open(logfile_PATH, 'a+') as f:
        f.write(f'Loaded ProteinSolver embeddings\n')  
        
################################
#### Concatenate embeddings ####
################################
        
#Concatenate PS and ESm embeddings
cat_embs_dict = dict()

# Iterate through sequence embeddings
for key, value in ESM_embs_dict.items():
    
    #print(f"Working with {count}/{len(ESM_embs_dict)}", end = "\r")
    
    #if structure embeddings exist - use it , else use zeros
    esm = value
    ps = PS_embs_dict[key]

    #Sanity check dimensions
    assert esm.shape == ps.shape
        
    #Concatenate the embeddings and add to dict
    Xs = torch.cat((esm,ps),1)
    cat_embs_dict[key] = Xs

#Update log        
with open(logfile_PATH, 'a+') as f:
        f.write(f'Concatenated embeddings\n')   

###################################
#### Finish up datapreparation ####
###################################
        
#Function that calculates amino acid distribution
def aa_dist(seq):
    counter = Counter(seq)
    aas = ["A","R","N","D","B","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","V"]
    dist = []
    for aa in aas:
        if aa in counter:
            dist.append(counter[aa]/len(seq))
        else:
            dist.append(0)
    return dist
    
#Ensure proper label/emb/clust order
data_labels = []
embs_X = []
clusters = []
count = 0
total = len(cat_embs_dict)
for key,embs in cat_embs_dict.items():
    count += 1
    row_num = df.loc[df['ID'] == key]
    label = [row_num.norm_NESG_label.item(),int(row_num.PSI_BIO_label)]
    
    #Also, add in the extra info
    template = [0] * len(embs)
    extra = aa_dist(row_num.sequence.item())
    extra_inf = extra + [len(row_num.sequence.item())]
    template = [extra_inf for x in template]
    extra_inf = torch.FloatTensor(template)
    
    #Append all in correct order
    clusters.append(int(row_num.cluster))
    data_labels.append(label)
    embs_X.append(torch.cat((embs,extra_inf), 1))
    
#Update log        
with open(logfile_PATH, 'a+') as f:
        f.write(f'Finished data preparation\n') 
        
##############################################        
#### Make functions for easier processing ####
##############################################
        
#Create dataset function
class ProteinDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        
        return (self.X[idx], torch.tensor(self.y[idx]))
        
#Create collate function for padding sequences
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0) 
    return xx_pad, yy
    
# Make model for saving models
def save_model(filepath, epoch, model, 
	           train_loss_values, train_psibio_AUC, train_nesg_PCC, train_psibio_MCC, train_labels,train_pred,
               val_loss_values, val_psibio_AUC, val_nesg_PCC, val_psibio_MCC, val_labels,val_pred,
               test_loss_values, test_psibio_AUC, test_nesg_PCC, test_psibio_MCC, test_labels,test_pred):
    
    #Save the trained model in various ways to ensure no loss of model
    
    #Create the folder
    isExist = os.path.exists(filepath)
    if not isExist:
        os.makedirs(filepath)

    ### METHOD 1 ###
    torch.save(model.state_dict(), filepath+"/model_conv.state_dict")

    #Later to restore:
    #model.load_state_dict(torch.load(filepath))
    #model.eval()

    ### METHOD 2 ###
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss' : train_loss_values,
        'train_AUC' : train_psibio_AUC,
        'train_PCC' : train_nesg_PCC,
        'train_MCC': train_psibio_MCC,
        'train_labels':train_labels,
        'train_pred': train_pred,
        'test_loss' : test_loss_values,
        'test_AUC' : test_psibio_AUC,
        'test_PCC' : test_nesg_PCC,
        'test_MCC': test_psibio_MCC,
        'test_labels':test_labels,
        'test_pred': test_pred,
        'val_loss' : val_loss_values,
        'val_AUC' : val_psibio_AUC,
        'val_PCC' : val_nesg_PCC,
        'val_MCC': val_psibio_MCC,
        'val_labels':val_labels,
        'val_pred': val_pred
    }

    torch.save(state, filepath+"/model_conv.state")

    #Later to restore:
    #model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])


    ### METHOD 3 ###
    torch.save(model, filepath+"/model_conv.full")

    #Later to restore:
    #model = torch.load(filepath)
    
    
#Make function for plotting performance         
def plot_performance(filepath,n_epochs,train_loss,val_loss,train_psibio_AUC,val_psibio_AUC,train_nesg_PCC,val_nesg_PCC,train_psibio_MCC,val_psibio_MCC):
    plt.ioff()
    
    plt.rcParams['figure.figsize'] = [20, 20]   
    plt.rcParams['font.size']=25

    #Initialize plot
    fig, (ax1, ax3, ax2, ax4) = plt.subplots(4, 1)
    fig.patch.set_facecolor('#FAFAFA')
    fig.patch.set_alpha(0.7)
    x = list(range(1,n_epochs+1))

    #Plot loss
    ax1.plot(x,train_loss, label = "Training_data", c="purple", lw = 3)
    ax1.plot(x,val_loss, label = "Validation_data", c="olive", lw = 3)
    #ax1.axhline(y=0, color='black',ls = "-" , lw=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training loss')
    ax1.set_title("Loss curve")
    ax1.legend(loc='upper center')
    ax1.set_xticks(np.arange(1,n_epochs,2))
    #ax1.set_yticks(np.arange(5*round(math.floor(min(val_loss)*100)/5)/100,math.ceil(max(train_loss)*100)/100,0.2))
    ax1.grid(True)

    #Plot accuracy
    #ax2.plot(x,train_nesg_ACC, label = "Training_nesg")
    #ax2.plot(x,val_nesg_ACC, label = "Validation_nesg")
    ax2.plot(x,train_psibio_AUC, label = "Training_psibio", lw = 3)
    ax2.plot(x,val_psibio_AUC, label = "Validation_psibio", lw = 3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title("AUC")
    ax2.legend(loc='lower center')
    ax2.set_xticks(np.arange(1,n_epochs,2))
    #ax2.set_yticks(np.arange(5*round(math.floor(min(val_psibio_AUC)*100)/5)/100,math.ceil(max(train_psibio_AUC)*100)/100,0.05))
    ax2.grid(True)
    
    #Plot correlation
    ax3.plot(x,train_nesg_PCC, label = "Training_nesg", lw = 3, c = "darkred")
    ax3.plot(x,val_nesg_PCC, label = "Validation_nesg", lw = 3, c = "gold")
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PCC')
    ax3.set_title("PCC")
    ax3.legend(loc='lower center')
    ax3.set_xticks(np.arange(1,n_epochs,2))
    #ax3.set_yticks(np.arange(5*round(math.floor(min(val_nesg_PCC)*100)/5)/100,math.ceil(max(train_nesg_PCC)*100)/100,0.05))
    ax3.grid(True)

    #Plot MCC
    #ax4.plot(x,train_nesg_MCC, label = "Training_nesg")
    #ax4.plot(x,val_nesg_MCC, label = "Validation_nesg")
    ax4.plot(x,train_psibio_MCC, label = "Training_psibio", lw = 3)
    ax4.plot(x,val_psibio_MCC, label = "Validation_psibio", lw = 3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('MCC')
    ax4.set_title("MCC")
    ax4.legend(loc='lower center')
    ax4.set_xticks(np.arange(1,n_epochs,2))
    #ax4.set_yticks(np.arange(5*round(math.floor(min(val_psibio_MCC)*100)/5)/100,math.ceil(max(train_psibio_MCC)*100)/100,0.05))
    ax4.grid(True)

    fig.tight_layout(pad = 1)
    filepath = f"{filepath}/Loss_ACC_MCC_{n_epochs}.png"
    fig.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none')
    
    plt.close(fig)

# Define my own group-K-fold splitter
def groupkfold(data_X, data_y, data_cluster, n=10):
    """Split the data for cross fold validation"""
    
    #initialize
    folds = []
    partitions = {}
    total_size = len(data_cluster)
    part_size = total_size/(n+1)
    cluster_dict = {}
    
    #unique clusters
    u_clust = set(data_cluster)
    count = 0
    for cluster in u_clust:
        count += 1
        temp = []
        for i, x in enumerate(data_cluster):
            if x == cluster:
                temp.append(i)
        cluster_dict[count] = temp
        
    #Sort clusters by size 
    s_clust = sorted(cluster_dict.items(), key=lambda x: len(x[1]),reverse=True)
    
    #Get test_idx 
    mid = int(len(s_clust)/2)
    get_list =  list(range(mid,mid+1001))
    test_idx = [part for i,(cl,part) in enumerate(s_clust) if i in get_list]
    test_idx = [x for sublist in test_idx for x in sublist]
    del s_clust[mid:mid+1001]
    
    #Split into partitions
    counter = 0
    skips = []
    for cl, part in s_clust:

        counter += 1
        if counter%(n+2) == 0:
            counter = 1
            
        while counter in skips:
            counter += 1
        
        if counter in partitions:
            if counter not in skips:
                partitions[counter] += part
                
                if len(partitions[counter]) > part_size:
                    skips.append(counter)
                
            else:
                print("Something went wrong")
                sys.exit(1)
        else:
            partitions[counter] = part

        
    #Double check that all is good
    tester_size = 0
    tester_sizes = []
    for p, part in partitions.items():
        tester_size += len(part)
        tester_sizes.append(len(part))
    if (max(tester_sizes)-min(tester_sizes)) > 150:
        print("Partitions does not match")
        print(f"Max: {max(tester_sizes)}, Min: {min(tester_sizes)}")
        sys.exit(1)
    elif len(partitions) != (n+1):
        print("The true length and needed lenght are not identical")
        sys.exit(1)
    
    #Make the folds
    ps = []
    partitions = sorted(partitions.items(), key=lambda x: x[0])
    
    #First get the test partition
    for p, part in enumerate(partitions):
        ps.append(p)
        val_idx = partitions[p][1]
        train_idx = [value[1] for i,value in enumerate(partitions) if i not in [p]]
        train_idx = [x for sublist in train_idx for x in sublist]
        folds.append([train_idx,val_idx,test_idx])
        
    return folds
            
#############################
#### START FROM HERE TAG ####
#############################

# split data
test_embs = embs_X
test_label = data_labels
test_clusters = clusters

#Alternative splits for testing
#test_embs = embs_X[1390:1470]
#test_label = data_labels[1390:1470]
#test_clusters = clusters[1390:1470]

#Perform partitioning
folds = groupkfold(test_embs, test_label, test_clusters, n=4)
foldperf = {}

#Hyper parameters
input_size = 60
hidden_size = 64
num_layers = 3
num_classes_nesg = 6 #7
num_classes_psibio = 2 #3
batch_size = 128 
n_epochs = 30 
lr = 0.01  #0.01
dropout = 0.4
weight_decay = 1e-6

#Define Bi_LSTM model

class Bi_LSTM(nn.Module) :
    def __init__(self, input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, num_classes_nesg = num_classes_nesg, num_classes_psibio = num_classes_psibio, dropout = dropout) :
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes_nesg = num_classes_nesg
        self.num_classes_psibio = num_classes_psibio
        self.dropout = dropout
            
        #Initialize the LSTM layer 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional = True, batch_first=True, dropout = dropout)
        
        #Initialize ReLU layer
        self.relu = nn.ReLU()
        
        #Initilize the linear layers for nesg labels 
        self.linear1 = nn.Linear((hidden_size * 2)+21, 1)
        
        #Initilize the linear layers for psibio labels
        self.linear2 = nn.Linear((hidden_size * 2)+21, num_classes_psibio)
        
        #Initialize softmax activation function 
        #self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        
        #Initialize the hidden state with random numbers
        self.hidden = (torch.randn(1, 1, self.hidden_size),torch.randn(1, 1, self.hidden_size))
        
        
    def forward(self, x):
        #Split embeddings and extra info for last dense layer
        embs, extra = torch.split(x, [60,21], dim=2)
        extra = torch.squeeze(extra)
        extra = extra.mean(1)
        #print(f"extra shape: {extra.shape}")
        #extra shape: torch.Size([128, 21])
        
        #batch normalize data
        self.bnorm = nn.BatchNorm1d(num_features=embs.shape[1])
        norm_data = self.bnorm(embs)
        norm_data.to(device)
      
        #Initialize the hidden states and cell states
        h_0 = torch.autograd.Variable(torch.zeros(self.num_layers*2, norm_data.size(0), self.hidden_size, device = norm_data.device)) #hidden state
        c_0 = torch.autograd.Variable(torch.zeros(self.num_layers*2, norm_data.size(0), self.hidden_size, device = norm_data.device)) #internal state
        h_0.to(device)
        c_0.to(device) 

        #forward through the lstm layer
        #print(f"initial shape: {norm_data.shape}")
        #initial shape: torch.Size([128, 804, 60])
        lstm_out,(ht, ct) = self.lstm(norm_data,(h_0, c_0))
        
        
        #concatenate states from both directions
        lstm_ht = torch.cat([ht[-1,:, :], ht[-2,:,:]], dim=1)
        #print(f"after lstm shape: {lstm_ht.shape}")
        #after lstm shape: torch.Size([128, 128])
        
        #Add the extra information before going through last dense layers
        collect = torch.cat((lstm_ht, extra), dim=1)
        
        #forward through relu layer
        #print(f"with_collection shape: {collect.shape}")
        #with_collection shape: torch.Size([128, 149])
        relu_nesg = self.relu(collect)
        relu_psibio = self.relu(collect)
        
        #forward through linear layer 1
        nesg_linear = self.linear1(relu_nesg)
        psibio_linear = self.linear2(relu_psibio)
        
        #Add sigmoid activation function
        sigmoid_nesg = self.sigmoid(nesg_linear)
        
        #Define output
        out1 = sigmoid_nesg
        out2 = psibio_linear

        return [out1, out2]

#Define the model, optimizer and loss function (removed  weight_decay = weight_decay) (removed weight=class_weights_nesg,)
model = Bi_LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, num_classes_nesg = num_classes_nesg, num_classes_psibio = num_classes_psibio, dropout = dropout)
model.to(device)
loss_nesg = nn.MSELoss(reduction='none')
loss_psibio = nn.CrossEntropyLoss(ignore_index=9, reduction = "mean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

#Train model function
def train_model(model, loss_func1, loss_func2, optimizer, n_epochs, fold):
    """Return trained model"""
    
    #Early stopping 
    patience = 3
    triggers = 0
    max_val_loss = 1000
    
    #Keep track of loss
    train_loss_values = []
    val_loss_values = []
    test_loss_values = []
    train_nesg_PCC = []
    train_psibio_MCC = []
    val_nesg_PCC = []
    val_psibio_MCC = []
    test_nesg_PCC = []
    test_psibio_MCC = []
    train_nesg_AUC = []
    train_psibio_AUC = []
    val_nesg_AUC = []
    val_psibio_AUC = []
    test_nesg_AUC = []
    test_psibio_AUC = []
   
     
    #Train network
    for epoch in range(1,n_epochs+1):
        
        #Keep track of train loss
        train_running_loss = 0.0
        train_nesg_pred = []
        train_psibio_pred = []
        train_nesg_labels = []
        train_psibio_labels = []
        
        #Keep track of val loss
        val_running_loss = 0.0
        val_nesg_pred = []
        val_psibio_pred = []
        val_nesg_labels = []
        val_psibio_labels = []
        
        #Keep track of test loss
        test_running_loss = 0.0
        test_nesg_pred = []
        test_psibio_pred = []
        test_nesg_labels = []
        test_psibio_labels = []
    
        model.to(device)
    
        #Iterate through batches
        for i, (embs, labels) in enumerate(train_loader):
            
            #reset optimizer
            optimizer.zero_grad()
            
            #Print sceen output
            str_epoch = format(epoch, '03d')
            str_batch = format(i+1, '03d')
            #print(f"Epoch: {str_epoch}, batch: {str_batch}", end="\r")     
            with open(logfile_PATH, 'a+') as f:
                f.write(f"Epoch: {str_epoch}, batch: {str_batch}\n")

            #Format labels 
            nesg_labels = torch.tensor([label[0] for label in labels])
            psibio_labels = torch.tensor([int(label[1]) for label in labels], dtype = torch.long)   
            nesg_labels.to(device)
            psibio_labels.to(device)
            embs.to(device)

 
            
            #Predict labels (forward)
            y_pred = model(embs)
            y_pred1 = torch.squeeze(y_pred[0])
            y_pred1.to(device)
            y_pred2 = y_pred[1]   
            y_pred2.to(device)
            
            #Make a mask vector
            multiply = torch.tensor([0 if x == 6 else 1 for x in nesg_labels])
            multiply.to(device)

            #Calculate MSE loss using masking
            loss1 = loss_func1(y_pred1, nesg_labels)
            non_zero_elements = multiply.sum()
            masked_loss = (loss1*multiply).sum()/non_zero_elements 
            
            #Calculate Cross Entropy loss
            loss2 = loss_func2(y_pred2, psibio_labels)
            
            #Combine loss (backward)
            combined_loss = masked_loss + loss2*0.25
            combined_loss.backward()
            train_running_loss += combined_loss.item() * embs.size(0)
            
            #optimize
            optimizer.step()
            
            #collect prediction and labels for comparison (overwrites every epoch)
            nesg = [pred.item() for pred in y_pred1]
            softmax = nn.Softmax(dim=0)
            psibio = [softmax(sublist)[1].item() for sublist in y_pred2]
            
            train_nesg_pred.append(nesg)
            train_psibio_pred.append(psibio)
            train_nesg_labels.append(nesg_labels.tolist())
            train_psibio_labels.append(psibio_labels.tolist())
            
        #Do the same for validation set
        model.eval()
        with torch.no_grad():
            for i, (embs,labels) in enumerate(val_loader):
                nesg_labels = torch.tensor([label[0] for label in labels])
                psibio_labels = torch.tensor([int(label[1]) for label in labels], dtype = torch.long)
                y_pred = model(embs)
                y_pred1 = torch.squeeze(y_pred[0])
                y_pred2 = y_pred[1]
                multiply = torch.tensor([0 if x == 6 else 1 for x in nesg_labels])
                loss1 = loss_func1(y_pred1, nesg_labels)
                non_zero_elements = multiply.sum()
                masked_loss = (loss1*multiply).sum()/non_zero_elements 
                loss2 = loss_func2(y_pred2, psibio_labels)
                combined_loss = masked_loss+loss2*0.25
                val_running_loss += combined_loss.item() * embs.size(0)
                nesg = [pred.item() for pred in y_pred1]
                softmax = nn.Softmax(dim=0)
                psibio = [softmax(sublist)[1].item() for sublist in y_pred2]
                val_nesg_pred.append(nesg)
                val_psibio_pred.append(psibio)
                val_nesg_labels.append(nesg_labels.tolist())
                val_psibio_labels.append(psibio_labels.tolist())
        model.train()    
        
        #Do the same for test set
        model.eval()
        with torch.no_grad():
            for i, (embs,labels) in enumerate(test_loader):
                nesg_labels = torch.tensor([label[0] for label in labels])
                psibio_labels = torch.tensor([int(label[1]) for label in labels], dtype = torch.long)
                y_pred = model(embs)
                y_pred1 = torch.squeeze(y_pred[0])
                y_pred2 = y_pred[1]
                multiply = torch.tensor([0 if x == 6 else 1 for x in nesg_labels])
                loss1 = loss_func1(y_pred1, nesg_labels)
                non_zero_elements = multiply.sum()
                masked_loss = (loss1*multiply).sum()/non_zero_elements 
                loss2 = loss_func2(y_pred2, psibio_labels)
                combined_loss = masked_loss+loss2*0.25
                test_running_loss += combined_loss.item() * embs.size(0)
                nesg = [pred.item() for pred in y_pred1]
                softmax = nn.Softmax(dim=0)
                psibio = [softmax(sublist)[1].item() for sublist in y_pred2]
                test_nesg_pred.append(nesg)
                test_psibio_pred.append(psibio)
                test_nesg_labels.append(nesg_labels.tolist())
                test_psibio_labels.append(psibio_labels.tolist())
        model.train() 
               
        #Collect loss after each epoch
        train_loss_values.append(train_running_loss / len(train_X))
        val_loss_values.append(val_running_loss / len(val_X))
        test_loss_values.append(test_running_loss / len(test_X))
        
        #Format predictions for MCC from batch lists to epoch lists
        train_nesg_pred = [item for sublist in train_nesg_pred for item in sublist]
        train_nesg_labels = [item for sublist in train_nesg_labels for item in sublist]
        train_psibio_pred = [item for sublist in train_psibio_pred for item in sublist]
        train_psibio_labels = [item for sublist in train_psibio_labels for item in sublist]
        val_nesg_pred = [item for sublist in val_nesg_pred for item in sublist]
        val_nesg_labels = [item for sublist in val_nesg_labels for item in sublist]
        val_psibio_pred = [item for sublist in val_psibio_pred for item in sublist]
        val_psibio_labels = [item for sublist in val_psibio_labels for item in sublist]
        test_nesg_pred = [item for sublist in test_nesg_pred for item in sublist]
        test_nesg_labels = [item for sublist in test_nesg_labels for item in sublist]
        test_psibio_pred = [item for sublist in test_psibio_pred for item in sublist]
        test_psibio_labels = [item for sublist in test_psibio_labels for item in sublist]
        
        #Make sample weights for MCC and ACC
        mcc_weight_train_psibio = [0 if lab == 9 else 1 for lab in train_psibio_labels]
        mcc_weight_val_psibio = [0 if lab == 9 else 1 for lab in val_psibio_labels]
        mcc_weight_test_psibio = [0 if lab == 9 else 1 for lab in test_psibio_labels]
        
        #Ignore indexes not needed for spearman correlation
        index_train_nesg = [i for i in range(len(train_nesg_labels)) if train_nesg_labels[i] != 6 ]
        index_val_nesg = [i for i in range(len(val_nesg_labels)) if val_nesg_labels[i] != 6 ]
        index_test_nesg = [i for i in range(len(test_nesg_labels)) if test_nesg_labels[i] != 6 ]
        new_train_nesg_labels = [train_nesg_labels[i] for i in index_train_nesg]
        new_train_nesg_pred = [train_nesg_pred[i] for i in index_train_nesg]
        new_val_nesg_labels = [val_nesg_labels[i] for i in index_val_nesg]
        new_val_nesg_pred = [val_nesg_pred[i] for i in index_val_nesg]
        new_test_nesg_labels = [test_nesg_labels[i] for i in index_test_nesg]
        new_test_nesg_pred = [test_nesg_pred[i] for i in index_test_nesg]
        
        #Ignore indexes not needed for AUC
        index_train_psibio = [i for i in range(len(train_psibio_labels)) if train_psibio_labels[i] != 9 ]
        index_val_psibio = [i for i in range(len(val_psibio_labels)) if val_psibio_labels[i] != 9 ]
        index_test_psibio = [i for i in range(len(test_psibio_labels)) if test_psibio_labels[i] != 9 ]
        new_train_psibio_labels = [train_psibio_labels[i] for i in index_train_psibio]
        new_train_psibio_pred = [train_psibio_pred[i] for i in index_train_psibio]
        new_val_psibio_labels = [val_psibio_labels[i] for i in index_val_psibio]
        new_val_psibio_pred = [val_psibio_pred[i] for i in index_val_psibio]
        new_test_psibio_labels = [test_psibio_labels[i] for i in index_test_psibio]
        new_test_psibio_pred = [test_psibio_pred[i] for i in index_test_psibio]
        
        #Format psibio labels for MCC
        mcc_train_pred = [round(x) for x in train_psibio_pred]
        mcc_val_pred = [round(x) for x in val_psibio_pred]
        mcc_test_pred = [round(x) for x in test_psibio_pred]
        
        #Collect MCC
        train_psibio_MCC.append(matthews_corrcoef(train_psibio_labels,mcc_train_pred,sample_weight=mcc_weight_train_psibio))
        val_psibio_MCC.append(matthews_corrcoef(val_psibio_labels, mcc_val_pred, sample_weight=mcc_weight_val_psibio))
        test_psibio_MCC.append(matthews_corrcoef(test_psibio_labels, mcc_test_pred, sample_weight=mcc_weight_test_psibio))
        
        #Collect pearson corelation
        train_correlation, p_value = spearmanr(new_train_nesg_labels, new_train_nesg_pred)
        train_nesg_PCC.append(train_correlation)
        val_correlation, p_value = spearmanr(new_val_nesg_labels, new_val_nesg_pred)
        val_nesg_PCC.append(val_correlation)
        test_correlation, p_value = spearmanr(new_test_nesg_labels, new_test_nesg_pred)
        test_nesg_PCC.append(test_correlation)
        
        #Collect AUC
        if len(set(new_train_psibio_labels))!= 1:
            train_AUC = roc_auc_score(new_train_psibio_labels, new_train_psibio_pred)
        else:
            train_AUC = 0
        if len(set(new_val_psibio_labels)) != 1:
            val_AUC = roc_auc_score(new_val_psibio_labels, new_val_psibio_pred)
        else:
            val_AUC = 0
        if len(set(new_test_psibio_labels)) != 1:
            test_AUC = roc_auc_score(new_test_psibio_labels, new_test_psibio_pred)
        else:
            test_AUC = 0
        train_psibio_AUC.append(train_AUC)
        val_psibio_AUC.append(val_AUC)
        test_psibio_AUC.append(test_AUC)
        
        #if epoch%2 == 0 or epoch == 1:
           # print(f"\nTraining loss: {train_running_loss / len(train_X)}\nValidation loss: {val_running_loss / len(test_X)}")
           # print(f"NESG PCC; Training: {train_correlation}  Validation: {val_correlation}")
           # print(f"PSI-BIO MCC; Training: {matthews_corrcoef(train_psibio_labels, mcc_train_pred,sample_weight=mcc_weight_train_psibio)}  Validation: {matthews_corrcoef(val_psibio_labels, mcc_val_pred, sample_weight=mcc_weight_val_psibio)}")
           # print(f"PSI-BIO AUC; Training: {train_AUC}  Validation: {val_AUC}")
            #print(f"\n")
    
            
        if epoch%1 == 0:
            filepath = f"{models_PATH}/fold{fold}/model_{epoch}_epochs"
            save_model(filepath, epoch, model, 
	           train_loss_values, train_psibio_AUC, train_nesg_PCC, train_psibio_MCC, [train_nesg_labels,train_psibio_labels],[train_nesg_pred,train_psibio_pred],
               val_loss_values, val_psibio_AUC, val_nesg_PCC, val_psibio_MCC, [val_nesg_labels,val_psibio_labels],[val_nesg_pred,val_psibio_pred],
               test_loss_values, test_psibio_AUC, test_nesg_PCC, test_psibio_MCC, [test_nesg_labels,test_psibio_labels],[test_nesg_pred,test_psibio_pred])
            plot_performance(filepath,epoch,train_loss_values, val_loss_values,train_psibio_AUC,val_psibio_AUC,train_nesg_PCC,val_nesg_PCC,train_psibio_MCC,val_psibio_MCC)
            
                
        #Check for early stopping
        current_loss = val_loss_values[-1]
        #print(f"Current: {current_loss}, biggest: {max_val_loss}, Triggers: {triggers}")
        if current_loss < max_val_loss:
            max_val_loss = current_loss
        else:
            triggers += 1
        if triggers == patience:
            return model, epoch, train_loss_values, train_nesg_PCC, train_psibio_MCC, train_psibio_AUC, val_loss_values, val_nesg_PCC, val_psibio_MCC, val_psibio_AUC,test_loss_values, test_nesg_PCC, test_psibio_MCC, test_psibio_AUC,[train_nesg_labels, train_psibio_labels, val_nesg_labels, val_psibio_labels,test_nesg_labels, test_psibio_labels],[train_nesg_pred, train_psibio_pred, val_nesg_pred, val_psibio_pred, test_nesg_pred, test_psibio_pred]   

    #Return model, loss values and MCC
    return model, epoch, train_loss_values, train_nesg_PCC, train_psibio_MCC, train_psibio_AUC,val_loss_values, val_nesg_PCC, val_psibio_MCC, val_psibio_AUC,test_loss_values, test_nesg_PCC, test_psibio_MCC, test_psibio_AUC,[train_nesg_labels, train_psibio_labels, val_nesg_labels, val_psibio_labels,test_nesg_labels, test_psibio_labels], [train_nesg_pred, train_psibio_pred, val_nesg_pred, val_psibio_pred, test_nesg_pred, test_psibio_pred]          
            
#Perform 5-fold cross validation
os.mkdir(models_PATH)
for fold, (train_idx,val_idx,test_idx) in enumerate(folds):
    start = time.time()
    fold = fold +1
    with open(logfile_PATH, 'a+') as f:
        f.write(f'####  Fold {fold}  ####\n')
        f.write(f"Train: {len(train_idx)}\n")
        f.write(f"val: {len(val_idx)}\n")
        f.write(f"Test: {len(test_idx)}\n")
      
    #Get the proper fold
    train_X, val_X, test_X = [test_embs[i] for i in train_idx], [test_embs[i] for i in val_idx], [test_embs[i] for i in test_idx]
    train_y, val_y, test_y = [test_label[i] for i in train_idx], [test_label[i] for i in val_idx], [test_label[i] for i in test_idx]

    
    #Make dataset
    train = ProteinDataset(train_X,train_y)
    val = ProteinDataset(val_X,val_y)
    test = ProteinDataset(test_X,test_y)
    
    #Make data loaders
    batch_size = 128 
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=pad_collate, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, collate_fn=pad_collate, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, collate_fn=pad_collate, shuffle=True)
    

    #Define model
    model = Bi_LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, num_classes_nesg = num_classes_nesg, num_classes_psibio = num_classes_psibio, dropout = dropout)
    model.to(device)
    loss_nesg = nn.MSELoss(reduction='none')
    loss_psibio = nn.CrossEntropyLoss(ignore_index=9, reduction = "mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

     
    #Train the model
    model, epoch, train_loss_values, train_nesg_PCC, train_psibio_MCC, train_psibio_AUC, val_loss_values, val_nesg_PCC, val_psibio_MCC, val_psibio_AUC, test_loss_values, test_nesg_PCC, test_psibio_MCC, test_psibio_AUC, labels_out, predictions_out = train_model(model,loss_nesg, loss_psibio, optimizer, n_epochs, fold)
                     
   
    #Save in a dictionary
    history = {
    	'epochs': epoch,
        'train_loss' : train_loss_values,
        'train_AUC' : train_psibio_AUC,
        'train_PCC' : train_nesg_PCC,
        'train_MCC': train_psibio_MCC,
        'test_loss' : test_loss_values,
        'test_AUC' : test_psibio_AUC,
        'test_PCC' : test_nesg_PCC,
        'test_MCC': test_psibio_MCC,
        'val_loss' : val_loss_values,
        'val_AUC' : val_psibio_AUC,
        'val_PCC' : val_nesg_PCC,
        'val_MCC': val_psibio_MCC,
        'labels' : labels_out,
        'predictions' : predictions_out
    }
    foldperf[f'fold{fold}'] = history
    end = time.time()

    
    #Screen output/logfile
    with open(logfile_PATH, 'a+') as f:
        f.write(f"---------------- Finished fold {fold} ----------------\n")
        f.write(f"Epoch: {epoch} \nTrain loss: {train_loss_values[-1]} \t Validation loss: {val_loss_values[-1]} \t Test loss: {test_loss_values[-1]}\n")
        f.write(f"Train AUC: {train_psibio_AUC[-1]} \t Validation AUC: {val_psibio_AUC[-1]} \t Test AUC: {test_psibio_AUC[-1]}\n")
        f.write(f"Train MCC: {train_psibio_MCC[-1]} \t Validation MCC: { val_psibio_MCC[-1]} \t Test MCC: {test_psibio_MCC[-1]}\n")
        f.write(f"Train PCC: {train_nesg_PCC[-1]} \t Validation PCC: { val_nesg_PCC[-1]} \t Test PCC: { test_nesg_PCC[-1]}\n")
        f.write("Elapsed time: {:.2f} min\n".format((end - start)/60))
        f.write("\n")

   # print(f"---------------- Finished fold {fold} ----------------")
   # print(f"Epoch: {total_epochs} \nTrain loss: {train_loss[-1]} \t Test loss: {val_loss[-1]}")
   # print(f"Train NESG PCC: {train_nesg_PCC[-1]} \t Test NESG PCC: {val_nesg_PCC[-1]}")
   # print(f"Train Psi-Bio MCC: {train_psibio_MCC[-1]} \t Test Psi-Bio MCC: { val_psibio_MCC[-1]}")
   # print(f"Train Psi-Bio AUC: {train_psibio_AUC[-1]} \t Test Psi-Bio AUC: { val_psibio_AUC[-1]}")
   # print("Elapsed time: {:.2f} min".format((end - start)/60))
   # print(f"--------------------------------------------------\n")
  
#Final save  
a_file = open(f"{models_PATH}/foldperf.pkl", "wb")
pickle.dump(foldperf, a_file)
a_file.close()


##################
#### Make plot ###
##################


##################
#### Make plot ###
##################
#Average output over epochs
import warnings
tlf, vlf, ttlf, tpcc, vpcc, ttpcc, tscc, vscc, ttscc, tr, vr, ttr = [],[],[],[],[],[],[],[],[],[],[],[]
max_epoch = 0
min_epoch = n_epochs
for n in range(n_epochs):
    tlfe, vlfe, ttlfe, tpcce, vpcce, ttpcce, tscce, vscce, ttscce, tre, vre, ttre = [],[],[],[],[],[],[],[],[],[],[],[]
    for f in range(1,6):
        try:
            max_epoch = (max(max_epoch,foldperf[f'fold{f}']['epochs']))
            min_epoch = (min(min_epoch,foldperf[f'fold{f}']['epochs']))
            tlfe.append(foldperf[f'fold{f}']['train_loss'][n])
            vlfe.append(foldperf[f'fold{f}']['val_loss'][n])
            ttlfe.append(foldperf[f'fold{f}']['test_loss'][n])
            tpcce.append(foldperf[f'fold{f}']['train_AUC'][n])
            vpcce.append(foldperf[f'fold{f}']['val_AUC'][n])
            ttpcce.append(foldperf[f'fold{f}']['test_AUC'][n])
            tscce.append(foldperf[f'fold{f}']['train_MCC'][n])
            vscce.append(foldperf[f'fold{f}']['val_MCC'][n])
            ttscce.append(foldperf[f'fold{f}']['test_MCC'][n])
            tre.append((foldperf[f'fold{f}']['train_PCC'][n]))
            vre.append((foldperf[f'fold{f}']['val_PCC'][n]))
            ttre.append((foldperf[f'fold{f}']['test_PCC'][n]))
            
            
        except IndexError as error:
            tlfe.append(np.nan)
            vlfe.append(np.nan)
            ttlfe.append(np.nan)
            tpcce.append(np.nan)
            vpcce.append(np.nan)
            ttpcce.append(np.nan)
            tscce.append(np.nan)
            vscce.append(np.nan)
            ttscce.append(np.nan)
            tre.append(np.nan)
            vre.append(np.nan)
            ttre.append(np.nan)
            foldperf[f'fold{f}']['train_loss'] = foldperf[f'fold{f}']['train_loss']+[np.nan]
            foldperf[f'fold{f}']['val_loss'] = foldperf[f'fold{f}']['val_loss']+[np.nan]
            foldperf[f'fold{f}']['test_loss'] = foldperf[f'fold{f}']['test_loss']+[np.nan]
            foldperf[f'fold{f}']['train_AUC'] = foldperf[f'fold{f}']['train_AUC']+[np.nan]
            foldperf[f'fold{f}']['val_AUC'] = foldperf[f'fold{f}']['val_AUC'] +[np.nan]
            foldperf[f'fold{f}']['test_AUC'] = foldperf[f'fold{f}']['test_AUC']+[np.nan]
            foldperf[f'fold{f}']['train_MCC'] = foldperf[f'fold{f}']['train_MCC']+[np.nan]
            foldperf[f'fold{f}']['val_MCC']  = foldperf[f'fold{f}']['val_MCC'] +[np.nan]
            foldperf[f'fold{f}']['test_MCC'] = foldperf[f'fold{f}']['test_MCC']+[np.nan]
            foldperf[f'fold{f}']['train_PCC']  = foldperf[f'fold{f}']['train_PCC'] +[np.nan]
            foldperf[f'fold{f}']['val_PCC']  = foldperf[f'fold{f}']['val_PCC'] +[np.nan]
            foldperf[f'fold{f}']['test_PCC']  = foldperf[f'fold{f}']['test_PCC'] +[np.nan]
            
    # Catch warnings for making means with nan        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tlf.append(np.nanmean(tlfe))
        vlf.append(np.nanmean(vlfe))
        ttlf.append(np.nanmean(ttlfe))
        tpcc.append(np.nanmean(tpcce))
        vpcc.append(np.nanmean(vpcce))
        ttpcc.append(np.nanmean(ttpcce))
        tscc.append(np.nanmean(tscce))
        vscc.append(np.nanmean(vscce))
        ttscc.append(np.nanmean(ttscce))
        tr.append(np.nanmean(tre))
        vr.append(np.nanmean(vre))
        ttr.append(np.nanmean(ttre))

#Get only the relevant length (max_epoch)        
tlf = tlf[:max_epoch]
vlf = vlf[:max_epoch]
ttlf = ttlf[:max_epoch]
tpcc = tpcc[:max_epoch]
vpcc = vpcc[:max_epoch]
ttpcc = ttpcc[:max_epoch]
tscc = tscc[:max_epoch]
vscc = vscc[:max_epoch]
ttscc = ttscc[:max_epoch]
tr = tr[:max_epoch]
vr = vr[:max_epoch]
ttr = ttr[:max_epoch]

for f in range(1,6):
    foldperf[f'fold{f}']['train_loss'] = foldperf[f'fold{f}']['train_loss'][:max_epoch]
    foldperf[f'fold{f}']['val_loss'] = foldperf[f'fold{f}']['val_loss'][:max_epoch]
    foldperf[f'fold{f}']['test_loss'] = foldperf[f'fold{f}']['test_loss'][:max_epoch]
    foldperf[f'fold{f}']['train_AUC'] = foldperf[f'fold{f}']['train_AUC'][:max_epoch]
    foldperf[f'fold{f}']['val_AUC'] = foldperf[f'fold{f}']['val_AUC'][:max_epoch]
    foldperf[f'fold{f}']['test_AUC'] = foldperf[f'fold{f}']['test_AUC'][:max_epoch]
    foldperf[f'fold{f}']['train_MCC'] = foldperf[f'fold{f}']['train_MCC'][:max_epoch]
    foldperf[f'fold{f}']['val_MCC']  = foldperf[f'fold{f}']['val_MCC'][:max_epoch]
    foldperf[f'fold{f}']['test_MCC'] = foldperf[f'fold{f}']['test_MCC'][:max_epoch]
    foldperf[f'fold{f}']['train_PCC']  = foldperf[f'fold{f}']['train_PCC'][:max_epoch]
    foldperf[f'fold{f}']['val_PCC']  = foldperf[f'fold{f}']['val_PCC'][:max_epoch]
    foldperf[f'fold{f}']['test_PCC']  = foldperf[f'fold{f}']['test_PCC'][:max_epoch]

    
#Make pretty plot
plt.rcParams['figure.figsize'] = [25, 25]   
plt.rcParams['font.size']=20

#Initialize plot
fig, ((ax1, ax3), (ax2,ax4)) = plt.subplots(2, 2)
fig.patch.set_facecolor('#FAFAFA')
fig.patch.set_alpha(0.7)
x = list(range(1,max_epoch+1))

#base * round(a_number/base)

###### Plot loss ######
ax1.plot(x,vlf, label = "Average loss of Validation data", c="blue", lw = 5 )
ax1.plot(x,tlf, label = "Average loss of Training data", c="red", lw = 5)
ax1.plot(x,ttlf, label = "Average loss of Testing data", c="cornflowerblue", lw = 5)
ax1.axvline(min_epoch,ls = '--', c = "grey", label = "First early stopping of a fold", lw = 3)

# Add each fold
ax1.plot(x,foldperf[f'fold1']['train_loss'], label = "Fold 1-5 Training data", c="palevioletred", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold2']['train_loss'], c="palevioletred", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold3']['train_loss'], c="palevioletred", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold4']['train_loss'], c="palevioletred", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold5']['train_loss'], c="palevioletred", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold1']['val_loss'], label = "Fold 1-5 Validation data", c="cornflowerblue", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold2']['val_loss'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold3']['val_loss'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold4']['val_loss'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold5']['val_loss'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold1']['test_loss'], label = "Fold 1-5 Test data", c="plum", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold2']['test_loss'], c="plum", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold3']['test_loss'], c="plum", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold4']['test_loss'], c="plum", lw = 3, alpha = 0.2)
ax1.plot(x,foldperf[f'fold5']['test_loss'], c="plum", lw = 3, alpha = 0.2)


# Make plot pretty
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training loss')
ax1.set_title("Loss curve")
ax1.legend(loc='upper center')
#ax1.set_xticks(np.arange(1,max_epoch+1,1))
#ax1.set_yticks(np.arange(0.26,0.37,0.02))
ax1.grid(True)

##### Plot AUC ####
ax2.plot(x,vpcc, label = "Average AUC of Validation Psi-Bio data", c = "blue", lw = 5)
ax2.plot(x,tpcc, label = "Average AUC of Training Psi-Bio data", c = "red", lw = 5)
ax2.plot(x,ttpcc, label = "Average AUC of Testing Psi-Bio data", c = "cornflowerblue", lw = 5)
ax2.axvline(min_epoch,ls = '--', c = "grey", label = "First early stopping of a fold", lw = 3)
# Add each fold
ax2.plot(x,foldperf[f'fold1']['train_AUC'], label = "Fold 1-5 Training data", c="palevioletred", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold2']['train_AUC'], c="palevioletred", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold3']['train_AUC'], c="palevioletred", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold4']['train_AUC'], c="palevioletred", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold5']['train_AUC'], c="palevioletred", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold1']['val_AUC'], label = "Fold 1-5 Validation data", c="cornflowerblue", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold2']['val_AUC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold3']['val_AUC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold4']['val_AUC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold5']['val_AUC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold1']['test_AUC'], label = "Fold 1-5 Test data", c="plum", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold2']['test_AUC'], c="plum", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold3']['test_AUC'], c="plum", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold4']['test_AUC'], c="plum", lw = 3, alpha = 0.2)
ax2.plot(x,foldperf[f'fold5']['test_AUC'], c="plum", lw = 3, alpha = 0.2)



# Make plot pretty
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title("AUC")
ax2.legend(loc='lower right')
#ax2.set_xticks(np.arange(1,max_epoch+1,1))
#ax2.set_yticks(np.arange(0.5,0.7,0.05))
ax2.grid(True)

    
#### Plot MCC ####
ax3.plot(x,vscc, label = "Average MCC of Validation Psi-Bio data", c = "blue", lw = 5)
ax3.plot(x,tscc, label = "Average MCC of Training Psi-Bio data", lw = 5, c="red")
ax3.plot(x,ttscc, label = "Average MCC of Testing Psi-Bio data", lw = 5, c="cornflowerblue")
ax3.axvline(min_epoch,ls = '--', c = "grey", label = "First early stopping of a fold", lw = 3)
# Add each fold
ax3.plot(x,foldperf[f'fold1']['train_MCC'], label = "Fold 1-5 Training data", c="palevioletred", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold2']['train_MCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold3']['train_MCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold4']['train_MCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold5']['train_MCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold1']['val_MCC'], label = "Fold 1-5 Validation data", c="cornflowerblue", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold2']['val_MCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold3']['val_MCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold4']['val_MCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold5']['val_MCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold1']['test_MCC'], label = "Fold 1-5 Test data", c="plum", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold2']['test_MCC'], c="plum", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold3']['test_MCC'], c="plum", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold4']['test_MCC'], c="plum", lw = 3, alpha = 0.2)
ax3.plot(x,foldperf[f'fold5']['test_MCC'], c="plum", lw = 3, alpha = 0.2)

# Make plot pretty
ax3.set_xlabel('Epoch')
ax3.set_ylabel('MCC')
ax3.set_title("MCC")
ax3.legend(loc='lower right')
#ax3.set_xticks(np.arange(1,max_epoch+1,1))
#ax3.set_yticks(np.arange(0,0.6,0.1))
ax3.grid(True)


#### Plot Accuracy ####
ax4.plot(x,vr, label = "Average SCC of Validation NESG data", c = "blue", lw = 5)
ax4.plot(x,tr, label = "Average SCC of Training NESG data", lw = 5, c="red")
ax4.plot(x,ttr, label = "Average SCC of Testing NESG data", lw = 5, c="cornflowerblue")
ax4.axvline(min_epoch,ls = '--', c = "grey", label = "First early stopping of a fold", lw = 3)
# Add each fold
ax4.plot(x,foldperf[f'fold1']['train_PCC'], label = "Fold 1-5 Training data", c="palevioletred", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold2']['train_PCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold3']['train_PCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold4']['train_PCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold5']['train_PCC'], c="palevioletred", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold1']['val_PCC'], label = "Fold 1-5 Validation data", c="cornflowerblue", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold2']['val_PCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold3']['val_PCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold4']['val_PCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold5']['val_PCC'], c="cornflowerblue", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold1']['test_PCC'], label = "Fold 1-5 Test data", c="plum", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold2']['test_PCC'], c="plum", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold3']['test_PCC'], c="plum", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold4']['test_PCC'], c="plum", lw = 3, alpha = 0.2)
ax4.plot(x,foldperf[f'fold5']['test_PCC'], c="plum", lw = 3, alpha = 0.2)


# Make plot pretty
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accurcay')
ax4.set_title("Accurcay")
ax4.legend(loc='lower right')
#ax4.set_xticks(np.arange(1,max_epoch+1,1))
#ax3.set_yticks(np.arange(0,0.6,0.1))
ax4.grid(True)

fig.tight_layout(pad = 1)
fig.savefig(f'{models_PATH}/Loss_ACC_MCC_pretty.png', facecolor=fig.get_facecolor(), edgecolor='none')
#plt.show()


#save_model("./final", n_epochs, model, train_loss, val_loss, train_nesg_PCC,  val_nesg_PCC, train_psibio_MCC, val_psibio_MCC, train_psibio_AUC, val_psibio_AUC, labels_out, predictions_out)   
#plot_performance("./final",n_epochs,train_loss, val_loss,train_psibio_AUC,val_psibio_AUC,train_nesg_PCC,val_nesg_PCC,train_psibio_MCC,val_psibio_MCC)        



