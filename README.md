# testing_evotrees
tests of the julia package evoTrees also considering integration with R.
- speed_comparison.jl was called with 4 threads set via the windows environment variable. xgboost and EvoTrees were called with 4 threads. EvoTrees took approximately 50% less time to train.

- speed_comparison.R calls the R package xgboost and EvoTrees via the package JuliaCall that allows communication with Julia from R. For 1000_000 features and 200 rounds, xgboost was quicker than evotrees. This is interpreted as overhead by JuliaCall. For 1500_000 features and 400 rounds, xgboost took 126 seconds and EvoTrees took 78 seconds for train. Hardly any difference in runtime for predict.

Next steps:
- A direct call of the R package [EvoTrees](https://github.com/Evovest/EvoTrees) did not work as the package could not be installed.
- Trying to call Julia with the package [JuliaConnectoR](https://github.com/stefan-m-lenz/JuliaConnectoR) to see whether it has less overhead than JuliaCall. This also works and can be replicated using the R-script speed_comparison_JuliaConnectoR.R. This solution however need a change to the EvoTrees package as the predict function was initially not exported ([see the discussion here](https://github.com/Evovest/EvoTrees.jl/pull/200)).

Summary: EvoTrees offers a speed-up of up to 50% (in pure Julia, less if called from R). JuliaCall could be used to call Julia from R.


