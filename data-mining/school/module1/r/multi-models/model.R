library(mlbench)
library(caret)
library(randomForest)
data(PimaIndiansDiabetes)


get_train_control <- function(type, no_fold, no_repeats) {
    control <- trainControl(method="repeatedcv", number=10, repeats=3) 
    return(control)
}

generate_seed <- function(n) {
    set.seed(n)
    return(TRUE)
}

train_model <- function(type, control, y_train, data_sets) {
    model <- train(y_train ~ ., data=data_sets, method=type, trContro=control)
    return(model)
}

CONTROL <- get_train_control("repeatedcv", 10, 3)
Y_TRAIN <- diabetes
DATA_SETS <- PimaIndiansDiabetes
# CART
generate_seed(7)
cart.model <- train_model("rpart", CONTROL, Y_TRAIN, DATA_SETS)

# LDA
generate_seed(7)
lda.model <- train_model("lda", CONTROL, Y_TRAIN, DATA_SETS)

# SVM
generate_seed(7)
svm.model <- train_model("svmRadial", CONTROL, Y_TRAIN, DATA_SETS)

#KNN
generate_seed(7)
knn.model <- train_model("knn", CONTROL, Y_TRAIN, DATA_SETS)

# Random Forest
fit.rf <- train(diabetes~., data=PimaIndiansDiabetes, method="rf", trControl=trainControl)
generate_seed(7)
rf.model <- train_model("rf", CONTROL, Y_TRAIN, DATA_SETS)

# collect resamples
results <- resamples(list(CART=cart.model, LDA=lda.model, SVM=svm.model, KNN=knn.model, RF=rf.model))

summary(results)

