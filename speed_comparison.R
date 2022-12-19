library(JuliaCall) ## notes on julia call https://hwborchers.github.io/
library(xgboost)

n_obs = 1000000
n_features = 100
nrounds = 200

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

julia_setup("C:\\Users\\rwarn\\AppData\\Local\\Programs\\Julia-1.8.2\\bin")
## https://cran.r-project.org/web/packages/JuliaCall/readme/README.html
## https://hwborchers.github.io/

## install the package if it is needed
julia_install_package_if_needed("EvoTrees")
## load julia package
julia_library("EvoTrees") 

## assign r objects to julia by reference
julia_assign("x_train", x_train)
julia_assign("y_train", y_train)

## Float64 
julia_eval("typeof(x_train)")
## can be converted to Float32
# julia_eval( "x_train = convert.(Float32,x_train);")

# EvoTrees params

## the max.depth parameter of EvoTrees corresponds to max.depth of xgboost-1
julia_command(paste0("params_evo = EvoTreeRegressor(
                          loss = :linear
                          ,nrounds=",nrounds,",alpha=0.5,lambda=0.0,gamma=0.0,eta=", params_xgb$eta,
                          ",max_depth=", params_xgb$max_depth+1, 
                          ",min_weight=",params_xgb$min_child_weight,
                          ",rowsample=", params_xgb$subsample,
                          ",colsample=", params_xgb$colsample_bytree,
                          ",nbins=", params_xgb$max_bin,
                          ",rng = 123)")
)

julia_assign("params_evo.device", "cpu")
julia_command("params_evo")

tictoc::tic()
julia_eval("m_evo = fit_evotree(params_evo; x_train, y_train, x_eval=x_train, y_eval=y_train, metric = :mae, print_every_n=50);")
print("\n")
tictoc::toc()

print("evotrees predict CPU:")

tictoc::tic()
pred_evo = julia_eval("EvoTrees.predict(m_evo, x_train)")
tictoc::toc()

## the predicted values r R-objects
length(pred_evo)

cor(pred_xgb, pred_evo)
