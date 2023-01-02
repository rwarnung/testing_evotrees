library(xgboost)

n_obs = 1500000 # 1500000
n_features = 100
nrounds = 400 #400

set.seed(20221224)

## generate train data
x_train = matrix(ncol= n_features, rnorm(n_obs*n_features))
y_train = rnorm(n_obs)

print(paste("train data generated with", n_features, "features and", n_obs, "observations"))

## xgboost 

print("xgboost train:")

params_xgb = list(
  booster = "gbtree", 
  eta = 0.05, 
  max_depth = 5, 
  min_child_weight = 1, 
  subsample = 0.5, 
  colsample_bytree = 0.5, 
  gamma = 0.0, 
  tree_method = "hist",
  objective = "reg:squarederror",
  max_bin = 64)

dtrain = xgb.DMatrix(x_train, label = y_train)

tictoc::tic()
xgb.model.tree = xgb.train(data = dtrain,  
                           params = params_xgb, nrounds = nrounds, verbose = 1, 
                           print_every_n = 50L) 
tictoc::toc()

print("xgboost predict:")
tictoc::tic()
pred_xgb = predict(xgb.model.tree, x_train);
tictoc::toc()

print("evotrees train:")

## using https://github.com/stefan-m-lenz/JuliaConnectoR
library(JuliaConnectoR)

# The package requires that Julia (version ??? 1.0) is installed and that the Julia executable is in the system search PATH 
# or that the JULIA_BINDIR environment variable is set to the bin directory of the Julia installation.
Sys.setenv(JULIA_BINDIR = "C:/Users/rwarn/AppData/Local/Programs/Julia-1.8.2/bin")

## check whether julia can be found
tryCatch(system2(command="julia", args="--version"))
EvoTrees = juliaImport("EvoTrees")
params_evo = EvoTrees$EvoTreeRegressor(nrounds = nrounds, alpha=0.5,lambda=0.0,gamma=0.0
                                       , eta=params_xgb$eta
                                       , max_depth = params_xgb$max_depth+1
                                       , min_weight = params_xgb$min_child_weight
                                       , rowsample = params_xgb$subsample
                                       , colsample = params_xgb$colsample_bytree
                                       , nbins = params_xgb$max_bin
                                       , device = "cpu"
                                       , loss = as.symbol("linear")
                                       , rng = 123L
                                       , T = juliaExpr("Float64")
                                       )

print("fields can be returned as follows")
params_evo$eta
params_evo$rng
print("values can be set as follows")
params_evo$eta = params_xgb$eta
#my_type = juliaExpr('Float64')
#params_evo$T = my_type  

tictoc::tic()
evoTrees_model = EvoTrees$fit_evotree(params_evo, x_train = x_train, y_train = y_train, print_every_n=50)
tictoc::toc()

print("evotrees predict CPU: TOOD Not sure how to do this. But speed improvement does not motivate this at the moment.")

