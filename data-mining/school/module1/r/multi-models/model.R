library(mlbench)
library(caret)
library(randomForest)
data(PimaIndiansDiabetes)


get_train_control <- function(type, no_fold, no_repeats) {
    control <- trainControl(method=type, number=no_fold, repeats=no_repeats) 
    return(control)
}

generate_seed <- function(n) {
    set.seed(n)
    return(TRUE)
}

train_model <- function(type, control, data_sets) {
    model <- train(diabetes~., data=data_sets, method=type, trControl=control)
    return(model)
}

get_params <- function() {
    return (list(data_sets=PimaIndiansDiabetes))
}

params <- get_params()
CONTROL <- get_train_control("repeatedcv", 10, 3)
DATA_SETS <- params$data_sets

# CART
generate_seed(7)
cart.model <- train_model("rpart", CONTROL, DATA_SETS)

# LDA
generate_seed(7)
lda.model <- train_model("lda", CONTROL, DATA_SETS)
# SVM
generate_seed(7)
svm.model <- train_model("svmRadial", CONTROL, DATA_SETS)
#KNN
generate_seed(7)
knn.model <- train_model("knn", CONTROL, DATA_SETS)
# Random Forest
generate_seed(7)
rf.model <- train_model("rf", CONTROL, DATA_SETS)

# collect resamples
results <- resamples(list(CART=cart.model, LDA=lda.model, SVM=svm.model, KNN=knn.model, RF=rf.model))
summary(results)
