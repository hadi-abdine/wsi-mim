from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
import time
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances as ed
import time
import sys
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = sys.float_info.epsilon

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def prepare_data(word_vectors, word_vectors2, val=None):
    data = [(word_vectors[i], word_vectors2[i]) for i in range(min(len(word_vectors2), len(word_vectors))) if not isinstance(word_vectors2[i], str) and  not isinstance(word_vectors[i], str) ]
    random.shuffle(data)
    if val is not None:
        X_train, X_val = train_test_split(data, test_size=val, random_state=42)
        return data, X_train, X_val
    else:
        return data, data, data

def prepare_test_data(word_vectors, word_vectors2):
    idxs = [i+1 for i in range(min(len(word_vectors2), len(word_vectors))) if not isinstance(word_vectors2[i], str) and  not isinstance(word_vectors[i], str) ]
    data = [(word_vectors[i-1], word_vectors2[i-1]) for i in idxs]
    return data, idxs


def IIC(z, zt, C=8):
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = ((P+P.t()) / 2) / P.sum()
    P[(P < EPS).data] = EPS
    Pi = P.sum(dim=1).view(C, 1).expand(C, C)
    Pj = P.sum(dim=0).view(1, C).expand(C, C)

    JM = (-0.1 / z.shape[0]) * torch.cdist(z, zt).sum(axis=0)[0]
    return (P * (torch.log(Pi) + torch.log(Pj) - torch.log(P))).sum()/z.shape[0] + JM, JM
    
class SenseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        content = self.data[index][0]
        content2 = self.data[index][1]
        sample = {'vector': content, 'vectort': content2}
        return sample

def collate_fn(data):
    vectors = torch.zeros(size=(len(data), len(data[0]['vector'])),
                               dtype=torch.float, device=device)
    vectorst = torch.zeros(size=(len(data), len(data[0]['vector'])),
                               dtype=torch.float, device=device)
    for i in range(len(data)):
        vector = data[i]['vector']
        vectort = data[i]['vectort']
        vectors[i] = torch.tensor(vector)
        vectorst[i] = torch.tensor(vectort)

    return vectors, vectorst

def get_loader(data, batch_size=5):
    """
    Args:
        path: path to dataset.
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader for custom dataset.
    """
    dataset = SenseDataset(data)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=collate_fn,
                                              drop_last=True,
                                              )

    return data_loader


class Model(nn.Module):
    def __init__(self, nbclasses, dim):
        super(Model, self).__init__()
        self.mlp1 = nn.Linear(dim, 2048)
        self.mlp2 = nn.Linear(2048, 2048)
        self.mlp3 = nn.Linear(2048, nbclasses)
        self.a = nn.Softmax()
        self.d = nn.Dropout(0.1)

        
    def forward(self, x):
        x = self.mlp1(x).to(device)
        x2 = F.relu(x)
        x2 = self.mlp2(x2)
        x3 =  F.relu(x2)
        x3 = self.mlp3(x3)
        probabilities = F.log_softmax(x3, dim=-1)
        probabilities = self.a(x3)
        return probabilities, x


def loadModel(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_validation_accuracy  = checkpoint['validation_accuracy']
    loss = checkpoint['loss']
    return epoch, best_validation_accuracy, loss

def trainModel(model, data, X_train, X_val, batch_size, training_epochs, C, device, word, checkpoints_path, printEvery=20, model_path=None, test=None, idxs=None, set=None, ids=None, eval_path=None, gold_path=None):
    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    optimizer = optim.AdamW(model.parameters(), lr=2e-5,
                                betas=(0.9, 0.999),
                                weight_decay=0.01)
    if device=='cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    if model_path != None:
        epoch, best_validation_accuracy, loss = loadModel(model,optimizer, model_path)

    time1 = time.time()
    training_accuracy_epochs = [] # save training accuracy for each epoch
    validation_accuracy_epochs = [] # save validation accuracy for each epoch 
    best_validation_loss = 1.0
    bestsc = 0
    loader_val = get_loader(X_val, len(X_val))
    for batch_val in loader_val:
        vectors1 = batch_val[0] # get input sequence of shape: batch_size * sequence_len
        vectors2 = batch_val[1] # get targets of shape : batch_size
    for i in range(epoch, training_epochs):
        print('-----EPOCH{}-----'.format(i+1))
        loader = get_loader(X_train, batch_size)
        
        for batch in loader:
            loss += trainOneBatch(model, batch, optimizer, C, scaler)
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                          time2 - time1, loss/printEvery))
                losses.append(loss)
                loss = 0 
        
        out1 = model.forward(vectors1)
        out2 = model.forward(vectors2)
        validation_loss = IIC(out1[0], out2[0], C)
        print('validation loss: ', validation_loss)
        best_validation_loss = min(best_validation_loss, validation_loss[0])
        if best_validation_loss == validation_loss[0]:
            print('saving checkpoint...')
            save_path = os.path.join(checkpoints_path, str(epoch))
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': validation_loss[0],
            'loss': loss,
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))


    epoch, best_validation_accuracy, loss = loadModel(model, optimizer, save_path)
    torch.save(model.state_dict(), os.path.join(checkpoints_path, word+'_model.pt')) 
    return best_validation_accuracy

def trainOneBatch(model, batch_input, optimizer, C, scaler):
    
    vectors = batch_input[0] # get input sequence of shape: batch_size * sequence_len
    vectors2 = batch_input[1] # get targets of shape : batch_size
    if device=='cuda':
        with torch.cuda.amp.autocast():
            out1 = model.forward(vectors)[0]
            out2 = model.forward(vectors2)[0]# shape: batch_size * number_classes
            loss = IIC(out1, out2, C)[0]
    else:
        out1 = model.forward(vectors)[0]
        out2 = model.forward(vectors2)[0]# shape: batch_size * number_classes
        loss = IIC(out1, out2, C)[0]
        
    optimizer.zero_grad()
    
    if device=='cuda':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward() # compute the gradient
        optimizer.step() # update network parameters
    return loss.item() # return loss value/


    
