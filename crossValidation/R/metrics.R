##' Root mean squared error
RMSE = function(pred, actual){
    sqrt(mean((pred-actual)^2))
}

##' Root mean squared logarithmic error
RMSLE = function(pred, actual){
    n = length(pred)
    sqrt(1/n * sum((log(pred + 1) - log(actual + 1))^2))
}