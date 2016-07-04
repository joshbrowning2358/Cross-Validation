#' Fit the Naive Bayes model
#' 
#' @param trainData The data.table containing the data for which the model 
#'   should be constructed.  We should have one or more rows for each 
#'   observation, and key will aggregate, across all observations, the binary 
#'   variables.
#' @param variables Vector of colnames of trainData which should be used to 
#'   construct the model.
#' @param targetVar The column name of trainData containing the target variable.
#' @param key Column name of trainData for identifying unique observations (and 
#'   there may be multiple observations for each key).
#' @param maxRatio Rare variables can cause extreme probabilities, as all
#'   observations may occur in one class.  To avoid this, we can constrain each
#'   variable to have a maximum difference between class probabilities (i.e.
#'   instead of predicting 100%/0% we prediction 99%/1%). This parameter
#'   enforces the max ratio between the two probabilities.
#' @param classification If TRUE, a naive bayes classifier is fit.  If FALSE, a
#'   Gaussian Naive Bayes model is used.
#' @return A model object to be passed to predictModel.
#'   

fitNaiveBayesModel = function(trainData, variables, targetVar="GENDER", key="localKey", maxRatio=Inf,
                              classification=TRUE){
    stopifnot(is(trainData, "data.table"))
    probabilityList = lapply(variables, function(var){
        levels = unlist(unique(trainData[, var, with = FALSE]))
        levels = levels[!is.na(levels)] # level may be NA, and that breaks things later with equals
        countsForVariable = lapply(levels, function(lvl){
            targetCounts = trainData[get(var) == lvl, length(unique(get(key))), by=c(targetVar)]
            targetCounts = dcast(. ~ get(targetVar), data=targetCounts, value.var = "V1")
            targetCounts[, "." := NULL]
            return(targetCounts)
        })
        countsDatatable = rbindlist(countsForVariable, fill=TRUE)
        data.table(variable = var, value = as.character(levels), counts = countsDatatable)
    })
    probabilities = do.call("rbind", probabilityList)
    targetCounts = trainData[, list(N=length(unique(get(key)))), by=c(targetVar)]
    for(i in 1:nrow(targetCounts)){
        target = targetCounts[i, get(targetVar)]
        # NA's get created in rbindlist, but they're really zeroes:
        probabilities[is.na(get(paste0("counts.", target))), c(paste0("counts.", target)) := 0]
        probabilities[, c(paste0("probability.", target)) := get(paste0("counts.", target)) / targetCounts[i, N]]
    }
    if(maxRatio < Inf){
        probabilityCols = grep("^probability\\.", colnames(probabilities), value=TRUE)
        probabilities[, maxProbability := apply(probabilities[, probabilityCols, with=FALSE], 1, max)]
        probabilities[, minAllowedProbability := maxProbability/maxRatio]
        for(i in 1:nrow(targetCounts)){
            target = targetCounts[i, get(targetVar)]
            probabilities[, c(paste0("probability.", target)) :=
                              pmax(get(paste0("probability.", target)), minAllowedProbability)]
        }
        probabilities[, c("maxProbability", "minAllowedProbability") := NULL]
    }
    return(probabilities)
}
