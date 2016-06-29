##' Check Cross Validation Model
##'
##' @param object An S4 object.
##' 
##' @return TRUE if there are no errors, otherwise the error messages.
##' 

checkCrossValidation = function(object){
    errors = character()
    if(object@extrapolationRange < 0){
        msg = "extrapolationRange can't be negative!"
        errors = c(errors, msg)
    }
    if(!object@level %in% c("global", "local")){
        msg = "level must be one of global or local."
        errors = c(errors, msg)
    }
    modelArguments = names(as.list(args(object@model)))
    if(object@level == "global"){
        requiredColumns = c("data", "imputationParameters")
        missing = requiredColumns[!requiredColumns %in% modelArguments]
        if(length(missing) > 1){
            msg = paste("model missing required arguments:",
                        paste(missing, collapse=", "))
            errors = c(errors, msg)
        }
    } else {
        # modelArguments should be the one argument and "", so length == 2
        if(length(modelArguments) != 2){
            msg = "Model should only contain one argument"
            errors = c(errors, msg)
        }
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
##' then set to NULL and use validationIndices.
##' 
##' \item{\code{validationIndices}:} A binary vector whose ith element is TRUE if
##' the ith row of xTrain should be used for the validation set.
##' 
##' @export crossValidation
##'   

ensembleModel = setClass(Class = "ensembleModel",
    representation = representation(model = "list",
                                    xTrain = "data.table",
                                    yTrain = "numeric",
                                    xTest = "data.table",
                                    cvIndices = "numeric",
                                    validationIndices = "numeric"),
    prototype(extrapolationRange = 0, level = "local"),
    validity = checkCrossValidation,
    package = "crossValidation")