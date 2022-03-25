
#% X: input matrix (nxp), scaled
#% y: outcome vector (nx1)
#% lambda: penalty parameter 
#% M: hierarchy parameter
#% batch_size
#% architecture: Feed forward or hierNet 
#% alpha0: initial step size/learning rate 

LassoNetR <- function(X, Y, NN, D_in ,
                      D_out, H, batch_size, lam,
                      M, set_seed = 42L, valid = None, SPLIT = 0.9, 
                      skip_bias = TRUE, n_epochs = 80L, alpha0 = 1e-3, 
                      optimizer = 'SGD', verbose = TRUE){
  
  X = reticulate::r_to_py(X)
  
  Y = reticulate::r_to_py(Y)
  # set up lassonet problem:
  prob <- module$lassonet_wrapper(X = X, Y = Y, NN = NN, D_in = D_in,
                                  D_out = D_out, H = H, batch_size=batch_size, lambda_= lam, M = M)
  
  
  return(prob)
}
