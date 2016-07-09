##' Root mean squared error
RMSE = function(pred, actual){
    sqrt(mean((pred-actual)^2))
}

##' Root mean squared logarithmic error
RMSLE = function(pred, actual){
    n = length(pred)
    sqrt(1/n * sum((log(pred + 1) - log(actual + 1))^2))
}

# AUC from Santander 3rd place team
AUC <-function (actual, predicted) {
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}