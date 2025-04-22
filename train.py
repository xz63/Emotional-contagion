import numpy as np
import math
import torch
from torch import nn
from model_downsample import *
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from AUdataset import *
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu') # if gpu is available use gpu

print(device)

scaler = MinMaxScaler()
# scaler = StandardScaler()
seq_len =  545 # should be 18 secs face action unit data
n_features = 17
AU = sio.loadmat(f'AUAll2.mat')
AU2=AU[f'AUAll2']
AU = sio.loadmat(f'AUAll1.mat')
AU1=AU[f'AUAll1']
print(AU[f'AUAll1'].shape)
print(AU1.shape)
temp=sio.loadmat(f'info12.mat');
info=temp[f'info12']

AU1 = scaler.fit_transform(AU1)
AU1 = scaler.fit_transform(AU1)
#AU = AU * 10.0
batch_size = 16

#### load data
#dataset_train = AUDataset(AU1,AU2 , chunk_length = seq_len, n_samples = 546) # emotion contagious
dataset_train = AUDataset(AU1,AU1 , chunk_length = seq_len, n_samples = 546) # auto encoding
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
############################3
# for j, (eeg_seq, fnirs_seq) in enumerate(val_loader):
#     print(j, eeg_seq.shape, fnirs_seq.shape)

enc_embedding_dim = 128
dec_embedding_dim = 128

dropout = 0.1

model = RecurrentAutoencoder(eeg_seq_len, fnirs_seq_len, eeg_features, fnirs_features, 
                        enc_embedding_dim, dec_embedding_dim, dropout, device)  # EEG to fNIRS
model = model.double() 
model = model.to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.MSELoss(reduction='mean').to(device)
epochs = 300

train_loss_all = []
val_loss_all = []

teacher_forcing_ratio = 0.5

for epoch in range(1, epochs+1):
    model = model.train()
    train_losses = []
    for j, (eeg_seq, fnirs_seq, eeg_phate_seq) in enumerate(train_loader):
        optimizer.zero_grad()            # no accumulation
        # print(eeg_seq.shape)
        eeg_seq = eeg_seq.to(device)   # putting sequence to gpu
        fnirs_seq = fnirs_seq.to(device)   # putting sequence to gpu
        eeg_phate_seq = eeg_phate_seq.to(device)
        # print(eeg_phate_seq.shape)
        seq_enc, seq_pred = model(eeg_seq, fnirs_seq, teacher_forcing_ratio)       # EEG to fNIRS prediction
        # seq_enc, seq_pred = model(fnirs_seq, eeg_seq, teacher_forcing_ratio)       # fNIRS to EEG prediction
        # print(eeg_seq[:, 1:,:].shape, seq_pred.shape)
        # print(seq_enc.shape)
        loss1 = criterion(seq_pred, fnirs_seq)  # measuring error
        # dist_ph = eeg_phate_seq[:, :eeg_seq_len -1, :]
        # loss2 = criterion()
        ph_dis = torch.cdist(eeg_phate_seq[:, :eeg_seq_len -1, :], eeg_phate_seq[:, 1:, :], p=2)
        latent_dis = torch.cdist(seq_enc[:, :eeg_seq_len -1, :], seq_enc[:, 1:, :], p=2)
        loss2 = criterion(ph_dis, latent_dis)
        loss = loss1  #+ loss2
        # loss = criterion(seq_pred, eeg_seq)  # measuring error
        loss.backward()                  # backprop
        optimizer.step()
        train_losses.append(loss.item())  # record loss by adding to training losses

    train_loss = np.mean(train_losses)   # computing loss on training and val data for this epoch
    print(f'Epoch {epoch}: train loss = {train_loss}')
    train_loss_all.append(train_loss)

    
    if epoch % 1 == 0:
        model = model.eval()
        val_losses = []
        with torch.no_grad():  # requesting pytorch to record any gradient for this block of code
            for k, (eeg_seq_true_ev, fnirs_seq_true_ev, eeg_phate_seq_true_ev) in enumerate(val_loader): 
                if k==0:
                    # print(eeg_seq_true_ev.shape)
                    eeg_seq_true_ev = eeg_seq_true_ev.to(device)   # putting sequence to gpu
                    fnirs_seq_true_ev = fnirs_seq_true_ev.to(device)   # putting sequence to gpu
                    eeg_phate_seq_true_ev = eeg_phate_seq_true_ev.to(device)
                    seq_enc, seq_pred_ev = model(eeg_seq_true_ev, fnirs_seq_true_ev, 0.1)       # EEG to fNIRS
                    # seq_enc, seq_pred_ev = model(fnirs_seq_true_ev, eeg_seq_true_ev, 0.01)       # fNIRS to EEG
                    loss1 = criterion(seq_pred_ev, fnirs_seq_true_ev)  # measuring error
                    # loss1 = criterion(seq_pred_ev[:, 1:, :], eeg_seq_true_ev[:, 1:, :])  # measuring error
                    val_losses.append(loss1.item())
                    break
                # print(seq_true_ev.shape)
            val_loss = np.mean(val_losses)
            print(f'Epoch {epoch}: Val loss = {val_loss}')
            val_loss_all.append(val_loss)
    
                

torch.save(model, f'./saved_Models/eeg2_fnirs134Decimated_downsample_{band}_Norm01.pth') #g2_3_eeg2fnirs_model sub4_eeg2fnirs_model
