# This file is adapted from here:
# https://rstudio.github.io/reticulate/articles/package.html

# global reference to LassoNet
module <- NULL

#.onLoad <- function(libname, pkgname) {
#  # use superassignment to update global reference to scipy
#  module <<- reticulate::import('module')

#}
library(reticulate)


install_lassonet <- function(method = "auto", conda = "auto") {
  # reticulate::py_install("lassonet", method = method, conda = conda, pip = TRUE)
  # vector of required python packages;
  #py_pack = c("numpy", "torch")
  #reticulate::py_install("module", "git+https://github.com/gnopuz83/lassonet@main", pip = T)
  module = import_from_path("module", path = "/Users/mara.stadler/LRZ Sync+Share/PhD/lassonet_fab_daniele",
                  convert = TRUE, delay_load = FALSE)
  
  #reticulate::py_install("git+https://github.com/gnopuz83/lassonet_fab@main", pip = T)
  
}

module = install_lassonet()