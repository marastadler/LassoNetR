hierNet versus LassoNet
================

Here we compare hierNet with weak hierarchy to LassoNet with a quadratic
hidden layer.

``` python
#% import python modules
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
```

``` python
from torch.optim.lr_scheduler import StepLR

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.metrics import mean_squared_error as rmse
from numpy import genfromtxt
```

\(y = 1x_1 + 2x_2 -3x_3 + 5x_5-1x_1x_2 +2x_5x_6\)

``` python
#% from prep_data import read_data, create_predicition_csv
#% import github repository lassonet reimplementation Fabian
path_lassonet = "/Users/mara.stadler/LRZ Sync+Share/PhD/lassonet_fab_daniele/"
import sys
sys.path.append(path_lassonet)
from module import LassoNet
from module import lassonet_wrapper
```

### Data simulation

``` r
#% Data simulation
D_in = 10 # input dimension
D_out = 1 # output dimension
H = 10 * 11 # hidden layer size

N = 1000 # training samples
batch_size = 15
```

``` r
set.seed(42)
X = matrix(rnorm(N * D_in), N, D_in)
dim(X)
```

    ## [1] 1000   10

``` r
y = 1 * X[, 1] + 2. * X[, 2] - 3*X[, 3] + 5*X[, 5] - 1.*X[, 1]*X[, 2] + 2*X[, 5]*X[,6]

X = scale(X)
```

``` python
class FeedForward(torch.nn.Module):
    """
    2-layer NN with RelU
    """

    def __init__(self, D_in, D_out, H):
        super().__init__()
        self.D_in = D_in
        self.D_out = D_out

        self.W1 = torch.nn.Linear(D_in, H, bias=True)
        self.relu = torch.nn.ReLU()
        self.W2 = torch.nn.Linear(H, H)
        self.W3 = torch.nn.Linear(H, D_out)
        return

    def forward(self, x):
        x = self.W1(x)
        x = self.relu(x)
        x = self.W2(x)
        x = self.relu(x)
        x = self.W3(x)
        return x
        
        
# %% Define HierNet
class myG(torch.nn.Module):
#     """
#     2-layer NN with RelU
#     """
     def __init__(self, D_in, D_out, H):
         super().__init__()
         self.D_in = D_in
         self.D_out = D_out
         self.W1 = torch.nn.Linear(D_in, D_in, bias = False)
         return

     def forward(self, x):
         # compute W^Tx
         y1 = torch.matmul(self.W1.weight.t(), x.t()).t()
         # compute Wx
         y2 = self.W1(x)
         y = (y2+y1)/2
         # compute x^T(W+W^T)/2 x
         z = torch.einsum('ij,ij->i',x,y).reshape(-1,1)
         return z
#Lassonet1 = lassonet_wrapper(X = XX_train, Y = yy_train, NN = myG, D_in = D_in,
#D_out = D_out, H = H, batch_size=batch_size, lambda_=0.1, M = 1.)
```

## hierNet with weak hierarchy

``` r
fitpath = hierNet::hierNet.path(x = X, y = y, strong = F)
fitcv = hierNet::hierNet.cv(fitpath, x = X, y = y)
# plot(fitcv)
```

``` r
fitcv$lamhat.1se
```

    ## [1] 53.22443

``` r
fitfinal = hierNet::hierNet(X, y, lam = fitcv$lamhat.1se)
```

    ## GG converged in 94 iterations.

``` r
fitfinal
```

    ## Call:
    ## hierNet::hierNet(x = X, y = y, lam = fitcv$lamhat.1se)
    ## 
    ## Non-zero coefficients:
    ##   (Rows are predictors with nonzero main effects)
    ##   (1st column is main effect)
    ##   (Next columns are nonzero interactions of row predictor)
    ##   (Last column indicates whether hierarchy constraint is tight.)
    ## 
    ##   Main effect 1       2       5 6      Tight?
    ## 1 0.9578      0       -0.9352 0 0            
    ## 2 1.9451      -0.9352 0       0 0            
    ## 3 -3.0402     0       0       0 0            
    ## 5 5.086       0       0       0 2.0615

OOS prediction `hierNet`

``` r
#X_valid = py$XX_valid_np
#y_valid = py$yy_valid_np
#y_valid = as.vector(y_valid)
#yhat <- predict(fitfinal, as.matrix(X_valid))

#rmse <- sqrt(mean((yhat - y_valid)^2))
#rmse
```

### LassoNet with linear + quadratic architecture

``` r
source("R/reticulate_setup.R")
source("R/LassoNetR.R")

fitlassonet <- LassoNetR(X = X, Y = y, NN = py$myG, D_in = 10L,
D_out = 1L, H = 10L, batch_size=15L, lam=0.1, M = 1L, n_epochs = 30L, valid = TRUE)
```

Training and validation loss

![](hierNet_lassoNet_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->
Linear weights

``` r
theta = fitlassonet$theta
theta = as.vector(theta)
names(theta) = paste0('x', 1:10)
round(theta, 2)
```

    ##    x1    x2    x3    x4    x5    x6    x7    x8    x9   x10 
    ##  1.00  1.99 -3.08  0.00  5.13 -0.03  0.00  0.00 -0.01  0.00

Quadratic weights (1st hidden layer)

``` r
library(RColorBrewer)
W1 = fitlassonet$W1
rownames(W1) = colnames(W1) = paste0('x', 1:10)
pheatmap::pheatmap(W1, cluster_rows = F, cluster_cols = F,
                   color = colorRampPalette(rev(brewer.pal(n = 7, name =
  "RdBu")))(100),
  breaks = seq(-max(abs(W1)), max(abs(W1)), length.out = 100),
  display_numbers = T)
```

![](hierNet_lassoNet_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
dim(W1)
```

    ## [1] 10 10

``` r
 #1.*X[:, 0] + 2.*X[:, 1] - 3*X[:, 2] + 5*X[:, 4] - 1.*X[:, 0]*X[:, 1] + 2*X[:, 4]*X[:,5]
```

### LassoNet with feed forward NN architecture

``` r
fitlassonet_FF <- LassoNetR(X = X, Y = y, NN = py$FeedForward, D_in = 10L,
D_out = 1L, H = 10L, batch_size=15L, lam=0.1, M = 1L, n_epochs = 30L, valid = T)
```

Training and validation loss

![](hierNet_lassoNet_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
thetaFF = fitlassonet_FF$theta
thetaFF = as.vector(thetaFF)
names(thetaFF) = paste0('x', 1:10)
round(thetaFF, 2)
```

    ##    x1    x2    x3    x4    x5    x6    x7    x8    x9   x10 
    ##  0.40  1.35 -2.70  0.12  3.30 -0.62 -0.11 -0.27 -0.20  0.08

``` r
W1FF = fitlassonet_FF$W1
colnames(W1FF) = paste0("x", 1:10)

pheatmap::pheatmap(W1FF, cluster_rows = F, cluster_cols = F,
                   color = colorRampPalette(rev(brewer.pal(n = 7, name =
  "RdBu")))(100),
  breaks = seq(-max(abs(W1FF)), max(abs(W1FF)), length.out = 100),
  display_numbers = T)
```

![](hierNet_lassoNet_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
 #1.*X[:, 0] + 2.*X[:, 1] - 3*X[:, 2] + 5*X[:, 4] - 1.*X[:, 0]*X[:, 1] + 2*X[:, 4]*X[:,5]
```

Now for a lambda and M path

``` r
lassonet_path = list()
lassonet_lam = list()
n_M = 0
M_all = c(1L, 2L, 3L, 5L, 10L, 50L, 100L, 1000L)
lam_all = c(0.00001, 0.001, 0.1, 0.5, 1L, 2L, 3L, 5L, 10L, 20L, 40L, 60L, 100L)
for(M in M_all){
  n_M = n_M + 1
  n_lam = 0

  for(lam in lam_all){
    

    n_lam = n_lam + 1
    lassonet_lam[[n_lam]] <- LassoNetR(X = X, Y = y, NN = py$FeedForward, 
                                    D_in = 10L, D_out = 1L, H = 10L, 
                                    batch_size=15L, 
                                    lam=lam, M = M, 
                                    n_epochs = 30L, valid = T)
   
  }
  #names(lassonet_path) <- paste0("lam = ", c(0.1, 0.5, 1L, 3L, 5L, 10L, 20L))
  lassonet_path[[n_M]] <- lassonet_lam
  names(lassonet_path)[[n_M]] <- paste0("M = ", M)
}
   
names(lassonet_path)
```

### Plot path training loss

``` r
n_epoch = 30
train_loss = matrix(nrow = length(M_all), ncol = length(lam_all))
valid_loss = matrix(nrow = length(M_all), ncol = length(lam_all))
for(m in 1:length(M_all)){
  for(l in 1:length(lam_all)){
    train_loss[m, l] = lassonet_path[[m]][[l]]$loss$train_loss[[n_epoch]]
    valid_loss[m, l] = lassonet_path[[m]][[l]]$loss$valid_loss[n_epoch]
  }
  
}
```

![](hierNet_lassoNet_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

![](hierNet_lassoNet_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

### Lassonet package with dense to sparse lambda path

``` python
from lassonet import LassoNetRegressor, plot_path
import matplotlib.pyplot as plt
```

``` python
modelLN = LassoNetRegressor(
    hidden_dims=(10,),
    eps_start=0.1,
    verbose=True,
)
#path = modelLN.path(XX_train, yy_train)

#plot_path(modelLN, path, XX_valid, yy_valid)
#plt.show()
```

``` python
#modelLN.feature_importances_.sort()

# feature_importances: the lambda value when that feature disappears

#modelLN.model.skip.weight.data

modelLN.load(path[100].state_dict).model.skip.weight.data
modelLN.load(path[300].state_dict).model.skip.weight.data
path.__len__()
```

Linear weights

``` r
betaLN = py$modelLN$feature_importances_$numpy()
names(betaLN) = paste0("x", 1:10)
betaLN
```

OOS prediction error

``` python
rmse(yy_valid.detach().numpy(), modelLN.predict(XX_valid).detach().numpy(), squared = False)
```

### DIABETES

``` python
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from lassonet import LassoNetRegressor, plot_path


dataset = load_diabetes()
X = dataset.data
y = dataset.target
_, true_features = X.shape
# add dummy feature
X = np.concatenate([X, np.random.randn(*X.shape)], axis=1)
feature_names = list(dataset.feature_names) + ["fake"] * true_features

# standardize
X = StandardScaler().fit_transform(X)
y = scale(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

modelD = LassoNetRegressor(
    hidden_dims=(10,),
    eps_start=0.1,
    verbose=True,
)
path = modelD.path(X_train, y_train)

plot_path(modelD, path, X_test, y_test)
```

``` python
#odelD.feature_importances_
#from itertools import islice
#islice(range(20), 1, 5)
modelD.feature_importances_
```

``` r
for(p in 1:10){
  py$modelD.load(path[-1].state_dict)
  py$modelD1.predict(X_train)
  print(sum(py$path[[p]]) != 0)
}
```

``` python
#path[1]
#modelD.score(X_train, y_train)
for i in range(len(path)):
    print(path[i].selected.sum(),i )
#modelD1 = modelD.load(path[-1].state_dict)
#modelD1.predict(X_train)
#modelD.path
#modelD.predict(X_test[2])
#modelD.predict(X)
```
