##' Root mean squared error
RMSE = function(pred, actual){
    sqrt(mean((pred-actual)^2))
}

##' Root mean squared logarithmic error
rmsle <- function(preds, actuals){
  n = length(preds)
  sqrt(1/n*sum((log(preds + 1) - log(actuals + 1))^2))
}

grad_rmsle <- function(preds, actuals){
  n = length(preds)
  rmsle_value = rmsle(preds, actuals)
  1/(2*rmsle_value) * 1/n * 2 * (log(preds + 1) - log(actuals + 1)) * 1/(preds + 1)
}

hess_rmsle <- function(preds, actuals){
  n = length(preds)
  rmsle_value = rmsle(preds, actuals)
  -1/(4*rmsle_value^3) * (1/n * 2 * (log(preds + 1) - log(actuals + 1)) * 1/(preds + 1))^2 +
    1/(2*rmsle_value) * 1/n * 2 * 1/(preds + 1)^2 * (1 + log(actuals + 1) - log(preds + 1))
}

# Test
# preds = 1:10
# actuals = preds + 0.1*rnorm(10)
# rmsle(preds, actuals)
# grad_rmsle(preds, actuals)
# numDeriv::grad(function(preds){rmsle(preds, actuals)}, x=preds)
# hess_rmsle(preds, actuals)
# diag(numDeriv::hessian(function(preds){rmsle(preds, actuals)}, x=preds))
# numDeriv::grad(function(x){grad_rmsle(c(x, preds[2:10]), actuals)[1]}, x=preds[1])


# AUC from Santander 3rd place team
AUC <-function (actual, predicted) {
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}