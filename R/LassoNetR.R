
#% X: input matrix (nxp), scaled
#% y: outcome vector (nx1)
#% lambda: penalty parameter 
#% M: hierarchy parameter
#% batch_size
#% architecture: Feed forward or hierNet 
#% alpha0: initial step size/learning rate 

LassoNetR <- function(X, Y, NN, D_in ,
                      D_out, H, batch_size, lam,
                      M, set_seed = 42L, valid = TRUE, SPLIT = 0.9, 
                      skip_bias = TRUE, n_epochs = 80L, alpha0 = 1e-3, 
                      optimizer = 'SGD', verbose = TRUE){
  
  X = reticulate::r_to_py(X)
  Y = reticulate::r_to_py(Y)
  valid = reticulate::r_to_py(valid)
  skip_bias = reticulate::r_to_py(skip_bias)
  verbose = reticulate::r_to_py(verbose)
  
  
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

