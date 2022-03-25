# This file is adapted from here:
# https://rstudio.github.io/reticulate/articles/package.html

# global reference to LassoNet
lassonet <- NULL

#.onLoad <- function(libname, pkgname) {
#  # use superassignment to update global reference to scipy
#  lassonet <<- reticulate::import("lassonet", delay_load = TRUE)

#}
library(reticulate)
install_lassonet <- function(method = "auto", conda = "auto") {
  # reticulate::py_install("lassonet", method = method, conda = conda, pip = TRUE)
  # vector of required python packages;
  #py_pack = c("numpy", "torch")
  module = import_from_path("module", path = "/Users/mara.stadler/LRZ Sync+Share/PhD/lassonet_fab_daniele/",
                   convert = TRUE, delay_load = FALSE)
  
  return(module)

}
module = install_lassonet()
