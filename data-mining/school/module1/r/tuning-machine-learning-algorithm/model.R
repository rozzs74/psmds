library(randomForest)
library(mlbench)
library(caret)

data(Sonar)

generate_seed <- function(n) {
    set.seed(n)
    return(TRUE)
}

get_data_sets <- function(data) {
    items <- list(raw=data, X=data[,1:60], Y=data[,61])
    return(items)
}

get_train_control <- function(type, no_fold, no_repeats, search_method) {
    control <- trainControl(method=type, number=no_fold, repeats=no_repeats, search=search_method) 
    return(control)
}

train_model <- function(type, control, data_sets, metric, tuneGrid) {
    model <- train(Class~., data=data_sets, method=type, trControl=control, metric=metric, tuneGrid=tuneGrid)
    return(model)
}

train_model_random_search <- function(type, control, data_sets, metric, tuneLength) {
    model <- train(Class~., data=data_sets, method=type, trControl=control, metric=metric, tuneLength=tuneLength)
    return(model)
}

train_model_with_ntree <- function(type, control, data_sets, metric, tuneGrid, ntree) {
    model <- train(Class~., data=data_sets, method=type, trainControl=control, metric=metric, tuneGrid=tuneGrid, ntree=ntree)
    return(model)
}

get_mtry <- function(value) {
    mtry <- sqrt(ncol(value))
    tunegrid <- expand.grid(.mtry=mtry)
    return(tunegrid)
}

normal_plot <- function(data) {
    plot(data)
    return(TRUE)
}

tune_random_forest <- function(x, y, factor, improvements, ntree) {
    optimal_mtry <- tuneRF(x, y, stepFactor=factor, improve=improvements, ntree=ntree)
}


dot_plot <- function (data) {
    dotplot(data)
    return(TRUE)
}
#RF
data.sets <- get_data_sets(Sonar)
generate_seed(7)
tune.grid <- get_mtry(data.sets$X)
model.name <- "rf"
model.metric <- "Accuracy"


# NORMAL
# CONTROL <- get_train_control("repeatedcv", 10, 3) #remove search on this function
# rf.model <- train_model(model.name, CONTROL, data.sets$raw, model.metric, tune.grid)

# RANDOM SEARCH
# CONTROL <- get_train_control("repeatedcv", 10, 3, "random")
# rf.model <- train_model_random_search(model.name, CONTROL, data.sets$raw, model.metric, 15)

# GRID_SEARCH
# CONTROL <- get_train_control("repeatedcv", 10, 3, "grid")
# rf.model <- train_model(model.name, CONTROL, data.sets$raw, model.metric, expand.grid(.mtry=c(1:15)))
# print(rf.model)

# normal_plot(rf.model)

# mtry <- tune_random_forest(data.sets$X, data.sets$Y, 1.5, 1e-5, 500)
# mtry


tune_manually <- function (ntrees) {
    CONTROL <- get_train_control("repeatedcv", 10, 3, "grid")
    tuneGrid <- expand.grid(.mtry=c(sqrt(ncol(data.sets$X))))
    models <- list()
    for (ntree in ntrees) {
        generate_seed(7)
        rf.model <- train_model_with_ntree("rf", CONTROL, data.sets$raw, model.metric, tuneGrid, ntree)
        key <- toString(ntree)
        models[[key]] <- rf.model
    }
    results <- resamples(models) 
    summary(results)
    dot_plot(results)
}

tune_manually(c(1000, 1500, 2000, 2500))



tune_via_extend_caret <- function() {
    customRF <- list(type="Classification", library="randomForest", loop=NULL)
    customRF$parameters <- data.frame(parameter=c("mtry", "ntree"), class=rep("numeric", 2),label=c("mtry", "ntree"))
    customRF$grid <- function(x, y, len=NULL, search="grid") { }

    customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
        randomForest(x, y, mtry=param$mtry, ntree=param$ntree, ...)
    }
    customRF$predict <- function(modelFit, newdata, preProc=NULL, submodels=NULL) {
        predict(modelFit, newdata)
    }

    customRF$prob <- function(modelFit, newdata, preProc=NULL, submodels=NULL) {
        predict(modelFit, newdata, type = "prob")
    }
    customRF$sort <- function(x) {
        x[order(x[,1]),]  
    }
    customRF$levels <- function(x) {
        x$classes
    }

    #note top bottom approach 

    trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
    tunegrid <- expand.grid(.mtry=c(1:15), .ntree=c(1000, 1500, 2000, 2500))
    set.seed(7)
    custom <- train(Class~., data=data.sets$raw, method=customRF, metric="Accuracy", tuneGrid=tunegrid, trControl=trainControl)
    summary(custom)
    plot(custom)
}

tune_via_extend_caret()