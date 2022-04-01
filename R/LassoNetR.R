
#' @param X input matrix (nxp), scaled
#' @param y outcome vector (nx1)
#' @param NN Feed forward or hierNet or own architecture
#' @param D_in NN input dimension
#' @param D_out NN output dimension
#' @param H dimension of the first hidden layer
#' @param batch_size batchsize optimizer
#' @param lam penalty parameter 
#' @param M hierarchy parameter
#' @param valid boolean(true split the data in validation and train set)
#' @param SPLIT size of testset
#' @param skip_bias boolean, default = TRUE
#' @param n_epochs number of epochs in NN
#' @param alpha0 initial step size/learning rate 
#' @param optimizer optimizer either 'SGD' or 'ADAM'
#' @param verbose should information be printed?, default: verbose = TRUE
#' @param step_size step size optimizer, stepLR parameter 
#' @param gamma stepLR parameter 


## Lassonet implementation: https://github.com/gnopuz83/lassonet.git

LassoNetR <- function(X, Y, NN, D_in ,
                      D_out, H, batch_size, lam,
                      M, set_seed = 42L, valid = TRUE, SPLIT = 0.9, 
                      skip_bias = TRUE, n_epochs = 80L, alpha0 = 1e-3, 
                      optimizer = 'SGD', verbose = TRUE, step_size = 20,
                      gamma = 0.7){

  X = reticulate::r_to_py(X)
  Y = reticulate::r_to_py(Y)
  D_in = reticulate::r_to_py(D_in)
  D_out = reticulate::r_to_py(D_out)
  H = reticulate::r_to_py(H)
  batch_size = reticulate::r_to_py(batch_size)
  lam = reticulate::r_to_py(lam)
  M = reticulate::r_to_py(M)
  set_seed = reticulate::r_to_py(set_seed)
  M = reticulate::r_to_py(M)
  valid = reticulate::r_to_py(valid)
  SPLIT = reticulate::r_to_py(SPLIT)
  skip_bias = reticulate::r_to_py(skip_bias)
  n_epochs = reticulate::r_to_py(n_epochs)
  alpha0 = reticulate::r_to_py(alpha0)
  verbose = reticulate::r_to_py(verbose)
  step_size = reticulate::r_to_py(step_size)
  gamma = reticulate::r_to_py(gamma)
  
  
  
  
  # set up lassonet problem:
  prob <- module$lassonet_wrapper(X = X, Y = Y, NN = NN, D_in = D_in,
                                  D_out = D_out, H = H, batch_size=batch_size, 
                                  lambda_= lam, M = M, set_seed = set_seed, 
                                  valid = valid, SPLIT = SPLIT, 
                                  skip_bias = skip_bias, n_epochs = n_epochs,
                                  alpha0 = alpha0, optimizer = optimizer, 
                                  verbose = verbose)
  
  
  return(prob)
}

