##' Check Cross Validation Model
##'
##' @param object An S4 object.
##' 
##' @return TRUE if there are no errors, otherwise the error messages.
##' 

checkCrossValidation = function(object){
    errors = character()
    if(any(names(object@model) != c("predict", "fit"))){
        msg = "model should be a list with elements predict and fit!"
        errors = c(errors, msg)
    }
    predictArguments = names(as.list(args(object@model$predict)))
    fitArguments = names(as.list(args(object@model$fit)))
    if(any(predictArguments != c("model", "X", ""))){
        msg = "model$predict should be a function with arguments model and X only!"
        errors = c(errors, msg)
    }    
    if(any(fitArguments != c("X", "y", ""))){
        msg = "model$fit should be a function with arguments X and y only!"
        errors = c(errors, msg)
    }
    if(length(object@cvIndices) > 0 & length(object@validationIndices) > 0){
        msg = "cvIndices and validationIndices cannot both have length > 0!"
        errors = c(errors, msg)
    }
    if(length(object@cvIndices) == 0 & length(object@validationIndices) == 0){
        msg = "cvIndices and validationIndices cannot both have length 0!"
        errors = c(errors, msg)
    }
    n = nrow(object@xTrain)
    if(length(object@cvIndices) != 0){
        if(length(object@cvIndices) != n){
            msg = "cvIndices must have length n!"
            errors = c(errors, msg)
        }
    }
    if(length(object@validationIndices) != 0){
        if(length(object@validationIndices) != n){
            msg = "validationIndices must have length n!"
            errors = c(errors, msg)
        }
    }
    if(length(object@yTrain) != n){
        msg = "yTrain must have length n!"
        errors = c(errors, msg)
    }
    if(length(errors) == 0)
        TRUE
    else
        errors
}

##' Cross Validation Class
##' 
##' \describe{
##' 
##' \item{\code{model}:}{The model to be applied to the data.  This should be a 
##' list with a fit function (taking just X and y) and a predict function 
##' (taking just a fitted model and X).}
##' 
##' \item{\code{xTrain}:} A data.table containing the training features.
##' 
##' \item{\code{yTrain}:} A vector containing the training target.
##' 
##' \item{\code{xTest}:} A data.table containing the testing features.
##' 
##' \item{\code{cvIndices}:} A vector describing which row belongs to which
##' cross-validation fold.  If you simply want to use one train/validation set,
##' then set to numeric(0) and use validationIndices.
##' 
##' \item{\code{validationIndices}:} A binary vector whose ith element is TRUE if
##' the ith row of xTrain should be used for the validation set.
##' 
##' @export crossValidation
##'   

crossValidation = setClass(Class = "crossValidation",
    representation = representation(model = "list",
                                    xTrain = "data.table",
                                    yTrain = "numeric",
                                    xTest = "data.table",
                                    cvIndices = "numeric",
                                    validationIndices = "logical"),
    validity = checkCrossValidation
#    package = "crossValidation"
    )

##' Summary method
##' 
##' Generates a summary of the crossValidation object.
##' 
##' @param object A crossValidation object.
##' 

setMethod("summary", "crossValidation", function(object){
    cat(sprintf("Training rows:           %s\n", nrow(object@xTrain)))
    cat(sprintf("Testing rows:            %s\n", nrow(object@xTest)))
    if(length(object@cvIndices) != 0)
        cat(sprintf("Cross-validation groups: %s\n", length(unique(object@cvIndices))))
    if(length(object@validationIndices) != 0){
        cat(sprintf("Train length:            %s\n", sum(!object@validationIndices)))
        cat(sprintf("Validation length:       %s\n", sum(object@validationIndices)))
    }
})

setMethod("print", "crossValidation", function(x){
    object = x
    cat(sprintf("Training rows:           %s\n", nrow(object@xTrain)))
    cat(sprintf("Testing rows:            %s\n", nrow(object@xTest)))
    if(length(object@cvIndices) != 0)
        cat(sprintf("Cross-validation groups: %s\n", length(unique(object@cvIndices))))
    if(length(object@validationIndices) != 0){
        cat(sprintf("Train length:            %s\n", sum(!object@validationIndices)))
        cat(sprintf("Validation length:       %s\n", sum(object@validationIndices)))
    }
})


##' Run the cross validation
##' 
##' Runs the cross validation, computes the score on the hold out set, and 
##' writes the results to a csv.
##' 
##' @param object A crossValidation object.
##' @param filename Where should the results be saved?
##' @param errorFunction A function accepting two equal length vectors and
##'   computing an error metric between them.
##' @param logged Should the output of this run be written to the log file?
##' 
##' @return Writes out a file with the results, prints the score.
##' 

setGeneric("run", function(object, filename=NULL, metric=RMSE, logged = FALSE)
    {standardGeneric("run")})
setMethod("run", signature(object="crossValidation"),
          function(object, filename=NULL, metric=RMSE, logged=FALSE){
    modelKey = as.numeric(Sys.time())
    if(is.null(filename)){
        filename = paste0(getwd(), "/", as.character(Sys.time(), "%Y%m%d%H%M%S"), "_", modelKey)
    } else {
        filename = gsub("\\.csv", paste0("_", modelKey), filename)
    }
    if(length(object@validationIndices) == 0){
        file = runCv(object, metric, filename)
    } else {
        file = runVal(object, metric, filename)
    }
    cat("Results of cross-validation saved in", file, "\n")
    if(!is.null(object@xTest)){
        file = runTrain(object, metric, filename)
        cat("Results of final prediction (fitting on full dataset) saved in", file, "\n")
    }
    if(logged){
        logResults(object, metric, filename, modelKey)
    }
})

##' Run CV with cross-validation
##' 
##' Helper function for run.
##' 
setGeneric("runCv", function(object, metric, filename){standardGeneric("runCv")})
setMethod("runCv", signature(object="crossValidation"), function(object, metric, filename){
    cvGroups = unique(object@cvIndices)
    predictions = rep(NA, length(object@yTrain))
    subScore = c()
    cat("Fitting model on", length(cvGroups), "cv groups...\n")
    for(i in cvGroups){
        filter = object@cvIndices != i
        model = object@model$fit(object@xTrain[filter, ], object@yTrain[filter])
        predictions[!filter] = object@model$predict(model, object@xTrain[!filter, ])
        subScore = c(subScore, metric(predictions[!filter], object@yTrain[!filter]))
        cat("Model for group", i, "completed with score", subScore[length(subScore)], "\n")
    }
    finalScore = metric(predictions, object@yTrain)
    cat("Modeling finished!  Final score: ", finalScore, "\n")
    filename = paste0(filename, "_", round(finalScore, 6), ".csv")
    write.csv(predictions, filename, row.names=FALSE)
    return(filename)
})

##' Run CV with validation
##' 
##' Helper function for run.
##' 
setGeneric("runVal", function(object, metric, filename){standardGeneric("runVal")})
setMethod("runVal", signature(object="crossValidation"), function(object, metric, filename){
    cat("Fitting model on ", sum(!object@validationIndices), " observations (",
        round(mean(!object@validationIndices), 2)*100, "%)\n", sep="")
    filter = !object@validationIndices
    model = object@model$fit(object@xTrain[filter, ], object@yTrain[filter])
    predictions = object@model$predict(model, object@xTrain[!filter, ])
    score = metric(predictions, object@yTrain[!filter])
    cat("Cross-validation finished!  Final score: ", score, "\n")
    filename = paste0(filename, "_", round(score, 6), ".csv")
    write.csv(predictions, filename, row.names=FALSE)
    return(filename)
})

##' Fit and analyze full training set
##' 
##' Helper function for run.
##' 
setGeneric("runTrain", function(object, metric, filename){standardGeneric("runTrain")})
setMethod("runTrain", signature(object="crossValidation"), function(object, metric, filename){
    cat("Fitting model on", nrow(object@xTrain), "observations\n")
    model = object@model$fit(object@xTrain, object@yTrain)
    predictions = object@model$predict(model, object@xTest)
    cat("Full model finished!\n")
    filename = paste0(filename, "_full.csv")
    write.csv(predictions, filename, row.names=FALSE)
    return(filename)
})

##' Log Results
##' 
##' Write the results of the analysis out to a log file.
##' 
setGeneric("logResults", function(object, metric, filename, modelKey){standardGeneric("logResults")})
setMethod("logResults", signature(object="crossValidation"),
          function(object, metric, filename, modelKey){
    log_file = "results_log.txt"
    
    functionText = as.character(body(object@model$fit))
    possibleModels = c("xgboost", "glm", "lm", "naiveBayes")
    match = sapply(possibleModels, function(mod){grep(mod, functionText)})
    suppressWarnings({
        matchIndex = min((1:length(match))[sapply(match, function(x){length(x) > 0})])
    })
    if(matchIndex == Inf){
        modelName = "Unidentified!"
        functionCall = paste(functionText, collapse="\n")
    } else {
        modelName = possibleModels[matchIndex]
        functionCall = functionText[unlist(match[matchIndex])]
    }
    
    sink(log_file, append=TRUE)
    try({
        cat("\n")
        cat("Environment:             R (", Sys.info()[4], ")\n", sep="")
        cat("Model Key:              ", modelKey, "\n")
        cat("File name:              ", filename, "\n")
        cat("Model name:             ", modelName, "\n")
        cat("Function call:          ", functionCall, "\n")
        cat("Execution time:         ", Sys.time(), "\n")
        summary(object)
    })
    sink()
})