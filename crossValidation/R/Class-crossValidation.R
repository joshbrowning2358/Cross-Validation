##' Check Cross Validation Model
##'
##' @param object An S4 object.
##' 
##' @return TRUE if there are no errors, otherwise the error messages.
##' 

checkCrossValidation = function(object){
    errors = character()
    if(names(object@model) != c("predict", "fit")){
        msg = "model should be a list with elements predict and fit!"
        errors = c(errors, msg)
    }
    predictArguments = names(as.list(args(object@model$predict)))
    fitArguments = names(as.list(args(object@model$predict)))
    if(predictArguments != c("X", "y", "")){
        msg = "model$predict should be a function with arguments X and y!"
        errors = c(errors, msg)
    }    
    if(fitArguments != c("X", "")){
        msg = "model$fit should be a function with argument X only!"
        errors = c(errors, msg)
    }
    if(is.null(cvIndices) & is.null(validationIndices)){
        msg = "cvIndices and validationIndices cannot both be NULL!"
        errors = c(errors, msg)
    }
    if(!is.null(cvIndices) & !is.null(validationIndices)){
        msg = "cvIndices and validationIndices cannot both be provided!"
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
##' then set to NULL and use validationIndices.
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
    validity = checkCrossValidation,
    package = "crossValidation")

setMethod("summary", "crossValidation", function(object){
    cat(sprintf("Training rows:           %s", nrow(xTrain)))
    cat(sprintf("Testing rows:            %s", nrow(xTest)))
    if(!is.null(cvIndices))
        cat(sprintf("Cross-validation groups: %s", length(unique(cvIndices))))
    if(!is.null(validationIndices)){
        cat(sprintf("Train length:            %s", sum(!validationIndices)))
        cat(sprintf("Validation length:       %s", sum(validationIndices)))
    }
})



