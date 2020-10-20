
library(mlbench)
library(caret)
library(caretEnsemble)

data(Ionosphere)


dataset <- Ionosphere
dataset <- dataset[, -2]
# change type factor to numeric
dataset$V1 <- as.numeric(as.character(dataset$V1))


trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"

boosting <- function() {
    #C5.0
    set.seed(seed)
    model.c50 <- train(Class ~., data=dataset, method="C5.0", metric=metric, trControl=trainControl)
    # model.c50
    #Stochastic Gradient Boosting
    set.seed(seed)
    model.gbm <- train(Class ~., data=dataset, method="gbm", metric=metric, trControl=trainControl)
    # model.gbm
    results <- resamples(list(c5.0=model.c50, gbm=model.gbm))
    summary(results)
    dotplot(results)
}



# boosting()


bagging <- function() {
    #treebag
    set.seed(seed)
    model.treebag <- train(Class ~., data=dataset, method="treebag", metric=metric, trControl=trainControl)
    # model.treebag
    set.seed(seed)
    #RF
    model.rf <- train(Class ~., data=dataset, method="rf", metric=metric, trControl=trainControl)
    # model.rf
    results <- resamples(list(treebag=model.treebag, rf=model.rf))
    summary(results)
    dotplot(results)
}

# bagging()


stacking <- function() {
    trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
    algorithms <- c("lda", "rpart", "glm", "knn", "svmRadial")
    set.seed(seed)
    models <- caretList(Class ~., data=dataset, trControl=trainControl, methodList=algorithms)
    # models
    results <- resamples(models)
    summary(results)
    dotplot(results)
    modelCor(results)
    splom(results)

    stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
    set.seed(seed)
    # #GLM
    stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl) 
    print(stack.glm)
    # #RF
    stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl) 
    print(stack.rf)
}
stacking()