from __future__ import print_function, division
import os
import numpy as np
import torch # pytorch package, allows using GPUs    
import pandas as pd
#seed fixe
seed=10
np.random.seed(seed)
torch.manual_seed(seed)
from torchvision import datasets
#importation des données du dataset SUSY 
class SUSY_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_file, root_dir, dataset_size, train=True, transform=None, high_level_feats=None):       
        #les 19 features qui sont représentées dans la database SUSY (la première est un booléen qui indique si la mesure est signal ou bruit)
        features=['SUSY','lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi', 
                'missing energy magnitude', 'missing energy phi', 'MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2', 
                'S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        #les "low_features" ,8 premières features, données brutes mesurées sur les collisions 
        low_features=['lepton 1 pT', 'lepton 1 eta', 'lepton 1 phi', 'lepton 2 pT', 'lepton 2 eta', 'lepton 2 phi', 
                'missing energy magnitude', 'missing energy phi']
        #les 10 suivantes "high_features" calculées qui sont connues comme importantes pour reconnaitre la supersymétrie , on les sépare des données brutes pour voir
        #si l'on peut se limiter à une partie du datastet
        high_features=['MET_rel', 'axial MET', 'M_R', 'M_TR_2', 'R', 'MT2','S_R', 'M_Delta_R', 'dPhi_r_b', 'cos(theta_r1)']
        #On lit le nombre demandé de données , Y le premier booléen , X les données mesurées
        file = pd.read_csv("SUSY.csv", header=None,nrows=dataset_size,engine='python')
        file.columns=features
        Y = file['SUSY']
        X = file[[col for col in file.columns if col!="SUSY"]]
        #On fixe le ratio train/test
        train_size=int(0.8*dataset_size)
        self.train=train
        #separation des dataset selon si l'on est en train ou en test
        if self.train:
            X=X[:train_size]
            Y=Y[:train_size]
            print("Training on {} examples".format(train_size))
        else:
            X=X[train_size:]
            Y=Y[train_size:]
            print("Testing on {} examples".format(dataset_size-train_size))


        self.root_dir = root_dir
        self.transform = transform

        #Pour utiliser une seule partie du datastet
        if high_level_feats is None:
            self.data=(X.values.astype(np.float32),Y.values.astype(int))
            print("high and low level features")
        elif high_level_feats is True:
            self.data=(X[high_features].values.astype(np.float32),Y.values.astype(int))
            print("high-level features")
        elif high_level_feats is False:
            self.data=(X[low_features].values.astype(np.float32),Y.values.astype(int))
            print("low-level features")
            
    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):

        sample=(self.data[0][idx,...],self.data[1][idx])

        if self.transform:
            sample=self.transform(sample)
        return sample
    
#Fix pour le dataloader très lent sous windows avec le gpu et workers torch
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
            
#loader de données pour mini batch train et test          
def load_data(args):
    data_file='SUSY.csv'
    root_dir=os.path.expanduser('~')+'\SUSY python'
    #arguments pour le GPU
    kwargs = {'pin_memory': True,'num_workers':4 ,'shuffle': True} 
    train_loader = FastDataLoader(SUSY_Dataset(data_file,root_dir,args.dataset_size,train=True,high_level_feats=args.high_level_feats),batch_size=args.batch_size,**kwargs)
    test_loader = FastDataLoader(SUSY_Dataset(data_file,root_dir,args.dataset_size,train=False,high_level_feats=args.high_level_feats),batch_size=args.test_batch_size,**kwargs)
    return train_loader, test_loader

#Création de la classe du réseau de neurones avec torch
import torch.nn as nn 
import torch.nn.functional as F 
class Net(nn.Module):
    def __init__(self,high_level_feats=None):
        super(Net, self).__init__()
        #Dimension des couches cachées du réseau
        dim1=40
        dim2=40
        dim3=40
        dim4=30
        #dim5=30
        #dimension de la première couche selon nombre de features utilisé
        if high_level_feats is None:
            self.fc1 = nn.Linear(18, dim1) 
        elif high_level_feats:
            self.fc1 = nn.Linear(10, dim1)
        else:
            self.fc1 = nn.Linear(8, dim1)

        #normalisation et creation des couches

        self.batchnorm1=nn.BatchNorm1d(dim1, eps=1e-05, momentum=0.1)
        self.batchnorm2=nn.BatchNorm1d(dim2, eps=1e-05, momentum=0.1)
        self.batchnorm3=nn.BatchNorm1d(dim3, eps=1e-05, momentum=0.1)
        self.batchnorm4=nn.BatchNorm1d(dim4, eps=1e-05, momentum=0.1)
        #self.batchnorm5=nn.BatchNorm1d(dim5, eps=1e-05, momentum=0.1)
        self.fc2 = nn.Linear(dim1, dim2)
        self.fc3 = nn.Linear(dim2, dim3)
        self.fc4 = nn.Linear(dim3, dim4)
        #self.fc5 = nn.Linear(dim4, dim5)
        self.fc5 = nn.Linear(dim4, 2)
    def forward(self, x):
        '''feed-forward function for the NN.
        x : autograd.Tensor
            input data
        Returns:
        autograd.Tensor
            output layer of NN

        '''

        #application de la fonction d'activation(relu)
        x = torch.relu(self.fc1(x))
        x = self.batchnorm1(x)
        #dropout
        #x = F.dropout(x, training=self.training)


        x = torch.relu(self.fc2(x))
        x = self.batchnorm2(x)
        #x = F.dropout(x, training=self.training)
        
        x = torch.relu(self.fc3(x))
        x = self.batchnorm3(x)
        #x = F.dropout(x, training=self.training)
        
        x = F.tanh(self.fc4(x))
        x = self.batchnorm4(x)
        #x = F.dropout(x, training=self.training)
        
        # x = F.relu(self.fc5(x))
        # x = self.batchnorm5(x)
        # x = F.dropout(x, training=self.training)
        
        x = self.fc5(x)
        #dernière couche softmax
        x = F.log_softmax(x,dim=1)

        return x

import torch.optim as optim 
def evaluate_model(args,train_loader,test_loader):

    #construction du réseau avec la classe Net
    DNN = Net(high_level_feats=args.high_level_feats).to(device)
    
    #definition de la fonction loss
    #criterion = F.nll_loss
    criterion = F.cross_entropy
    
    #definition de l'optimiseur descente de gradian 
    optimizer = optim.SGD(DNN.parameters(), lr=0.01, momentum=args.momentum)
    #optimizer = optim.Adam(DNN.parameters(), lr=args.lr, betas=(0.9, 0.999))
    #optimizer = optim.AdamW(DNN.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    #optimizer = optim.Adadelta(DNN.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.0)
    def train(epoch):
        '''Trains a NN with batches.

        Parameters
        ----------
        epoch : int
            Training epoch number.

        '''

        DNN.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            data,label = data.to(device), label.to(device)
            optimizer.zero_grad()
            #calcul de l'output sur train set
            output = DNN(data)
            label = label.type(torch.LongTensor)
            #calcul de la train loss
            data,label,output = data.to(device), label.to(device),output.to(device)
            loss = criterion(output,label).to(device)
            #algorithme de backpropagation
            loss.backward()
            #mise à jour des poids
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() ))
            

        return loss.item()

    def test():
        '''Tests NN performance.

        '''

        DNN.eval()
        test_loss = 0 
        correct = 0 
        for data, label in test_loader:
            data,label = data.to(device), label.to(device)
            #calcul de l'output sur test set
            output = DNN(data)
            #calcul de la test loss
            label = label.type(torch.LongTensor)
            data,label,output = data.to(device), label.to(device),output.to(device)
            test_loss += criterion(output,label,size_average=False).item()
            #calcul de la loss la plus probable
            pred = output.data.max(1, keepdim=True)[1]
            #nombre de prédiction correctes
            correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset))) 
        
        return test_loss, correct / len(test_loader.dataset)
    train_loss=np.zeros((args.epochs,))
    test_loss=np.zeros_like(train_loss)
    test_accuracy=np.zeros_like(train_loss)
    epochs=range(1, args.epochs + 1)
    #boucle sur les epoch
    for epoch in epochs:
        train_loss[epoch-1] = train(epoch)
        test_loss[epoch-1], test_accuracy[epoch-1] = test()
    return test_loss[-1], test_accuracy[-1]
    train_loss=np.zeros((args.epochs,))
    test_loss=np.zeros_like(train_loss)
    test_accuracy=np.zeros_like(train_loss)
    epochs=range(1, args.epochs + 1)
    
    for epoch in epochs:
        train_loss[epoch-1] = train(epoch)
        test_loss[epoch-1], test_accuracy[epoch-1] = test()
    return test_loss[-1], test_accuracy[-1]

#code plot  
import matplotlib.pyplot as plt

def plot_data(x,y,data):
    fontsize=16
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)    
    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])  
            ax.text(x_val, y_val, c, va='center', ha='center')
    x=[str(i) for i in x]
    y=[str(i) for i in y]
    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)
    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{dataset\\ size}$',fontsize=fontsize)
    plt.tight_layout()
    plt.show()
    
def grid_search(args):
    #on essaie avec différentes tailles de données et learning rates
    dataset_sizes=[1000,10000,100000,200000,500000,1000000]
    #batch_sizes=1000
    #learning_rates=np.logspace(-3,-1,3)
    # dataset_sizes=[1000000]
    learning_rates= [0.01]
    test_loss=np.zeros((len(dataset_sizes),len(learning_rates)),dtype=np.float64)
    test_accuracy=np.zeros_like(test_loss)
    for i, dataset_size in enumerate(dataset_sizes):
        args.dataset_size=dataset_size
        #args.batch_size=batch_sizes
        args.batch_size=int(0.01*dataset_size)
        train_loader, test_loader = load_data(args)
        for j, lr in enumerate(learning_rates):
            args.lr=lr
            print("\n training DNN with %5d data points lr=%0.6f. \n" %(dataset_size,lr) )
            test_loss[i,j],test_accuracy[i,j] = evaluate_model(args,train_loader,test_loader)
    plot_data(learning_rates,dataset_sizes,test_accuracy)
   
import argparse 
parser = argparse.ArgumentParser(description='PyTorch SUSY Example')
parser.add_argument('--dataset_size', type=int, default=100000, metavar='DS',
                help='size of data set (default: 100000)')
parser.add_argument('--high_level_feats', type=bool, default=None, metavar='HLF',
                help='toggles high level features (default: None)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                help='input batch size for training (default: 100)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                help='learning rate (default: 0.02)')
parser.add_argument('--momentum', type=float, default=0.8, metavar='M',
                help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                help='disables CUDA training')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                help='how many batches to wait before logging training status')
args = parser.parse_args()
#set device pour que le code tourne sur les ordinateurs sans gpu (tf32 pour les gpu rtx 30)
if not args.no_cuda and torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    device = torch.device('cpu')
#model = Net().to(device) 
#device = torch.device('cpu')
#seed
torch.manual_seed(args.seed)
import time
start = time.time()
#name main pour que le code s'execute correctement sous windows
if __name__=='__main__':
    grid_search(args)
end = time.time()
#mesure du temps
print("execution time = {}".format(end-start))

