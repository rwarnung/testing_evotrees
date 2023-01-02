

params_evo$eta

#############


n_obs = 100 # 1500000
n_features = 100
nrounds = 10L #400

set.seed(20221224)

x_train = matrix(ncol= n_features, rnorm(n_obs*n_features))
y_train = rnorm(n_obs)

library(JuliaConnectoR)
evoTrees = juliaImport("EvoTrees")
params_evo = evoTrees$EvoTreeRegressor(nrounds=nrounds , loss = as.symbol("linear"), alpha=0.5,lambda=0.0,gamma=0.0)

evoTrees_model = evoTrees$fit_evotree(params_evo, x_train, y_train, print_every_n = 50L)

juliaCall("typeof", y_train)
