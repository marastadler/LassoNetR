# LassoNetR

`LassoNetR` can be used to solve a deep neural network with feature sparsity in R. 
For this we follow the idea of  Lemhadri et al. (2019) with the corresponding python module 
`lassonet` (https://github.com/lasso-net/lassonet).

Internally, the package uses the solver from an adaptation of the lassonet module in python, 
decribed in this paper: https://arxiv.org/abs/1907.12207.

- arbitrary feed-forward networks
- optimal lambda selection via stability selection or cross-validation
- lambda path (dense-to-sparse??)

