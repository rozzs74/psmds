#############################################
# Author: John Royce C. Punay               #
# Date created: October 3, 2020, 11:43 AM   #
#############################################

library(mlbench)
library(caret)
data(Ionosphere)

get_data_sets <- function() {
    return(Ionosphere)
}

data_sets <- get_data_sets()
# data_sets[1,]

get_data_set_dimension <- function(data_sets) {
	dim(data_sets)
}

get_data_set_columns <- function(data_sets) {
	names(data_sets)
}

get_data_sets_data_types <- function(data_sets) {
	sapply(data_sets, class)
}

# get_data_set_dimension(data_sets)

# get_data_set_columns(data_sets)

# get_data_sets_data_types(data_sets)



get_knn_params <- function() {
	return (list(method="knn", train_method="repeateadcv", train_number=10, train_repeats=3, seed_value=7, model_evaluation_metric="Accuracy"))
}
params <- get_knn_params()

data_sets$V1 <- as.numeric(as.character(data_sets$V1))
# sapply(data_sets, class)

set_seed <- function(n) {
	set.seed(n)
	return(TRUE)
}
train_method <- params$train_method
train_number <- params$train_number
train_repeats <- params$train_repeats
knn.train.control <- trainControl(method=train_method, number=train_number, repeats=train_repeats)

set_seed(7)
train_knn_model <- function(data_sets, controls, method, metric) {
	knn_model <- train(Class ~., data=data_sets, method=method, metric=metric, trControl=control)
	return(knn_model)
}
algorithm <- params$method
algorithm_metric <- params$model_evaluation_metric
knn_model <- train_knn_model(data_sets, knn.train.control, algorithm, algorithm_metric)
knn_model
summary(knn_model)
