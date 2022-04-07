import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from collections import OrderedDict
from typing import Callable

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import warnings



# The following code is copied (with small adaptations) from https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py
# Copyright of Louis Abraham, Ismael Lemhadri


def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)

def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)

def hier_prox(v, u, lambda_, lambda_bar, M):
    """
    v has shape (k,) or (k, d)
    u has shape (K,) or (K, d)
    
    standard case described in the paper: v has size (1,d), u has size (K,d)
    
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values
    k, d = u.shape
    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, d).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)
    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)
    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, d)
    w_star = torch.gather(w, 0, idx).view(1, d)
    
    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star

#%% own implementation of LassoNet

class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx, :]
        y = self.targets[idx]
        return x, y
    




class LassoNet(torch.nn.Module):
    def __init__(self, G: torch.nn.Module, lambda_: float=0.01, M: float=10, skip_bias: bool=False):
        """
        Implementation of LassoNet for arbitrary architecture. See https://jmlr.org/papers/volume22/20-848/20-848.pdf for details.

        Parameters
        ----------
        G : ``torch.nn.Module``
            The nonlinear part of LassoNet. Needs the following attributes:
                * ``self.W1`` : the linear layer applied to the inputs. This is called W^(1) in the LassoNet paper.
                * ``self.D_in`` : dimension of input
                * ``self.D_out`` : dimension of output
        lambda_ : float, optional
            Penalty parameter for the skip layer. The default is 1.
        M : float, optional
            Penalty parameter for the hierarchical constraint. The default is 1.
        skip_bias : boolean, optional
            Whether the skip connection has a bias.
        
        Returns
        -------
        None.

        """
        super().__init__()
        
        self.G = G
        self.lambda_ = lambda_
        self.M = M
        self.D_in = self.G.D_in
        self.D_out = self.G.D_out
        
        self.skip = torch.nn.Linear(self.D_in, self.D_out, bias = skip_bias) # skip connection aka theta
        return
    
    def forward(self, x):
        y1 = self.G(x)
        y2 = self.skip(x)
        return y1+y2
    
    def train_epoch(self, loss: torch.nn.Module, dl: DataLoader, opt: torch.optim.Optimizer=None, preprocess: Callable=None):
        """
        Trains one epoch.
        
        Parameters
        ----------
        loss : ``torch.nn`` loss function
            Loss function for the model.
        dl : ``torch.utils.data.DataLoader``
            DataLoader with the training data.
        opt : from ``torch.optim.Optimizer``, optional
            Pytorch optimizer. The default is SGD with Nesterov momentum and learning rate 0.001.
        preprocess : function, optional
            A function for preprocessing the inputs for the model. The default is None.
        
        Returns
        -------
        info : dict
            Training loss and accuracy history. 

        """
        if opt is None:
            opt = torch.optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, nesterov = True)
        
        info = {'train_loss':[],'train_acc':[]}
                    
        ################### START OF EPOCH ###################
        self.train()
        for inputs, targets in dl:
            if preprocess is not None:
                inputs = preprocess(inputs)
            
            # forward pass
            y_pred = self.forward(inputs)
            # compute loss
            loss_val = loss(y_pred, targets)           
            # zero gradients
            opt.zero_grad()    
            # backward pass
            loss_val.backward()    
            # iteration
            opt.step()
            # step size
            alpha = opt.state_dict()['param_groups'][0]['lr']
            # prox step
            self.skip.weight.data, self.G.W1.weight.data = hier_prox(self.skip.weight.data, self.G.W1.weight.data,\
                                                                        lambda_=self.lambda_*alpha, lambda_bar=0, M = self.M)
            
            ## COMPUTE ACCURACY AND STORE 
            _, predictions = torch.max(y_pred.data, 1)
            accuracy = (predictions == targets).float().mean().item()
            info['train_loss'].append(loss_val.item())
            info['train_acc'].append(accuracy)
          
            
                    
        return info
      




  
  

def lassonet_wrapper(X, Y, NN, lambda_, M, D_in, D_out, H, batch_size, set_seed = 42, valid = None, SPLIT = 0.9, skip_bias = True, n_epochs = 80, alpha0 = 1e-3, optimizer = 'SGD', verbose = True):
      
    '''
  NN the architecture of the neural network
  
  X:Scaled input nxp
  Y:target nx1
  valid:boolean(true split the data in validation and train set)
  SPLIT:splitting parameter
  
  D_in:NN input dimension
  D_out:NN output dimension
  H:dimension of the first hidden layer
  batch_size:batch_size
  n_epochs:number of epochs of NN
  alpha0:initial step size learning rate
    '''
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)
    if valid:
        x, x_valid, y, y_valid = train_test_split(X,Y, train_size = SPLIT, random_state = set_seed)
        ds = MyDataset(x,y)
        dl = DataLoader(ds, batch_size = batch_size, shuffle = True)
        valid_ds = MyDataset(x_valid, y_valid)
    
    else:
        ds = MyDataset(X,Y)
        dl = DataLoader(ds, batch_size = batch_size, shuffle = True)
    
  # neural network creation
  
  
    G = NN(D_in = D_in, D_out = D_out, H = H)
  
    model = LassoNet(G, lambda_ = lambda_, M = M, skip_bias = skip_bias)
    loss = torch.nn.MSELoss(reduction='mean')


    
    if optimizer == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
    elif optimizer == 'ADAM':
        opt = torch.optim.Adam(model.parameters(), lr = alpha0)
    else:
        warnings.warn("optimizer has to be SGD or ADAM, change it to SGD")
        opt = torch.optim.SGD(model.parameters(), lr = alpha0, momentum = 0.9, nesterov = True)
    
    lr_schedule = StepLR(opt, step_size = 20, gamma = 0.7)

    
    if valid:
        loss_hist = {'train_loss':[], 'valid_loss':[]}
    else:
        loss_hist = {'train_loss':[]}  
    
    
    for j in np.arange(n_epochs): 
        if verbose:
            print(f"================== Epoch {j+1}/{n_epochs} ================== ")
    #print(opt)  
    
    ### TRAINING
        epoch_info = model.train_epoch(loss, dl, opt=opt)
        loss_hist['train_loss'].append(np.mean(epoch_info['train_loss']))
    
        if lr_schedule is not None:
            lr_schedule.step()
    
    ### VALIDATION
        if valid:
            model.eval()
            output = model.forward(valid_ds.data)          
            valid_loss = loss(output, valid_ds.targets).item()
            loss_hist['valid_loss'].append(valid_loss)
            if verbose:
                print(f"  validation loss: {valid_loss}.")    
        if verbose:
            print(f"  train loss: {np.mean(epoch_info['train_loss'])}.")
  
    final_return = {
        'model':model, 'NN': G, 'theta' : model.skip.weight.data.numpy(), 'W1' : G.W1.weight.data.numpy()}

    
    return final_return
      


class Create_own_NN(torch.nn.Module):

    def __init__(self, D_in, D_out, H, activation = False):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.H = H
        self.activation = activation
        self.hier = False
        
        if self.activation:
            if type(self.activation) == type(OrderedDict()):
                self.network = activation
            else:
                self.from_2_list_to_network(activation)
                
            #
        else:
            self.network = OrderedDict([
          ('lin1', torch.nn.Linear(D_in, H, bias = True)),
          ('relu1', torch.nn.ReLU()),
          ('lin2', torch.nn.Linear(H, H)),
          ('lin3', torch.nn.Linear(H, D_out))
        ])
            
        self.net = torch.nn.Sequential(self.network)
        self.W1 = self.network['lin1']
       
    
    def forward(self, X):
        if self.hier:
            y = F.linear(X, self.W1.weight)
            z = (x*y).sum(1)[:None]
            return(z)
        else:
            one_step = self.net(X)
            self.W1 = self.net[0]
            return one_step
    
    def from_2_list_to_network(self, lista):
        
        assert len(lista) == 2, 'it should be a list with 2 lists'
        
        my_net = OrderedDict()
        
        lin = 1
        relu1 = 1
        sigmoid1 = 1
        matm = 1
        unknown_l = 1
        for el1, el2 in zip(lista[0], lista[1]):
            if el1 == 'lin':
                name_l = el1+str(lin)
                lin += 1
                
                my_net[name_l] = torch.nn.Linear(el2[0], el2[1], bias = el2[2])
            elif el1 == 'relu':
                name_r = el1+str(relu1)
                relu1 += 1
                my_net[name_r] = torch.nn.ReLU()
            
            elif el1 == 'sig':
                name_s = el1+str(relu1)
                sigmoid1 += 1
                my_net[name_s] = torch.nn.Sigmoid()
                
            elif el1 == 'mat':
                name_m = el1+str(matm)
                matm += 1
                self.hier = True
            else:
                name_u = el1+str(unknown_l)
                unknown_l += 1
                my_net[name_u] = el2
                
        
        self.network = my_net

