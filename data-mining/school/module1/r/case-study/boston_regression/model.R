library(mlbench)
library(caret)
library(corrplot)
library(Cubist)

data(BostonHousing)

# Split out validation dataset
# create a list of 80% of the rows in the original dataset we can use for training
set.seed(7)
validationIndex <- createDataPartition(BostonHousing$medv, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- BostonHousing[-validationIndex,]
# use the remaining 80% of data to training and testing the models
dataset <- BostonHousing[validationIndex,]

dataset$chas <- as.numeric(as.factor(dataset$chas))

dim(dataset)
sapply(dataset, class)

# summary(dataset)
# cor(dataset[,1:13])

# quartz("Test")
# par(mfrow=c(2,7))
# for(i in 1:13) {
#   hist(dataset[,i], main=names(dataset)[i])
# }

# par(mfrow=c(2,7))
# for(i in 1:13) {
#   plot(density(dataset[,i]), main=names(dataset)[i])
# }

# par(mfrow=c(2,7))
# for(i in 1:13) {
#   boxplot(dataset[,i], main=names(dataset)[i])
# }


# #scatterplot matrix
# pairs(dataset[,1:13])

# # correlation plot
correlations <- cor(dataset[,1:13]) 
# corrplot(correlations, method="circle")

run_linear_algorithms <- function(dataset, metric, preProc) {
	#LM
	set.seed(7)
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	fit.lm <- train(medv~., data=dataset, method="lm", metric=metric, preProc=preProc, trControl=trainControl)
	#GLM
	set.seed(7)
	fit.glm <- train(medv~., data=dataset, method="glm", metric=metric, preProc=preProc, trControl=trainControl)
	#GLMNET
	set.seed(7)
	fit.glmnet <- train(medv~., data=dataset, method="glmnet", metric=metric, preProc=preProc, trControl=trainControl)
	return(list(lm=fit.lm, glm=fit.glm, glmnet=fit.glmnet))
}

tune_svm <- function (dataset, metric, preProc) {
	#Tune SVM sigman and C parameters
	grid <- expand.grid(.sigma=c(0.025, 0.05, 0.15), .C=seq(1, 10, by=1))
	set.seed(7)
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric, preProc=preProc, trControl=trainControl, tuneGrid=grid)
	print(fit.svm)
	return(fit.svm)	
}

run_non_linear_algorithms <- function(dataset, metric, preProc) {
	#SVM
	set.seed(7)
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric, preProc=preProc, trControl=trainControl)

	#CART
	set.seed(7)
	grid <- expand.grid(.cp=c(0, 0.05, 0.1))
	fit.cart <- train(medv~., data=dataset, method="rpart", metric=metric, tuneGrid=grid, preProc=preProc, trControl=trainControl)

	#KNN
	set.seed(7)
	fit.knn <- train(medv~., data=dataset, method="knn", metric=metric, preProc=preProc, trControl=trainControl)
	return(list(svm=fit.svm, cart=fit.cart, knn=fit.knn))
}



run_feature_selection <- function(cutoff) {
	# Feature selection
	# remove correlated attributes
	# find the attributes that are highly correlated
	set.seed(7)
	highlyCorrelatedIndexes <- findCorrelation(correlations, cutoff=cutoff)
	for(value in highlyCorrelatedIndexes) {
		print(names(dataset)[value])
	}
	#create a new dataset without highly corrected features
	datasetFeatures <- dataset[, -highlyCorrelatedIndexes]
	# dim(datasetFeatures)
	# head(datasetFeatures)
	return(datasetFeatures)
}



# new_data_set <- run_feature_selection(0.70)
# linears <- run_linear_algorithms(new_data_set, "RMSE", c("center", "scale", "BoxCox"))
# non_linears <- run_non_linear_algorithms(new_data_set, "RMSE", c("center", "scale", "BoxCox"))
# feature_results <- resamples(list(LM=linears$lm, GLM=linears$glm, GLMNET=linears$glmnet, SVM=non_linears$svm, CART=non_linears$cart, KNN=non_linears$knn))
# summary(feature_results)
# dotplot(feature_results)


# svm <- tune_svm(new_data_set, "RMSE", c("BoxCox"))
# plot(svm)


run_esemble_methods <- function(dataset, metric, preProc) {
	# try ensembles
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)

	#Random Forest
	set.seed(7)
	fit.rf <- train(medv~., data=dataset, method="rf", metric=metric, preProc=preProc, trControl=trainControl)

	#Stochastic Gradient Boosting
	set.seed(7)
	fit.gbm <- train(medv~., data=dataset, method="gbm", metric=metric, preProc=preProc, trControl=trainControl, verbose=FALSE)

	#Cubist
	set.seed(7)
	fit.cubist <- train(medv~., data=dataset, method="cubist", metric=metric, preProc=preProc, trControl=trainControl)
	return(list(rf=fit.rf, gbm=fit.gbm, cubist=fit.cubist))
}

# ensembleResults <- run_esemble_methods(dataset, "RMSE", c("BoxCox"))
# finalResults <- resamples(list(RF=ensembleResults$rf, GBM=ensembleResults$gbm, CUBIST=ensembleResults$cubist))
# summary(finalResults)
# dotplot(finalResults)

tune_cubist <- function(dataset, metric, preProc) {
	# Tune the Cubist algorithm
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	set.seed(7)
	grid <- expand.grid(.committees=seq(15, 25, by=1), .neighbors=c(3, 5, 7)) 
	tune.cubist <- train(medv~., data=dataset, method="cubist", metric=metric, preProc=preProc, tuneGrid=grid, trControl=trainControl)
	return(tune.cubist)
}

# cubist <- tune_cubist(dataset, "RMSE", c("BoxCox"))
# print(cubist)
# plot(cubist)

finalize_model <- function(dataset) {
	set.seed(7)
	x <- dataset[,1:13]
	y <- dataset[,14]
	preprocessParams <- preProcess(x, method=c("BoxCox"))
	transX <- predict(preprocessParams, x)
	finalModel <- cubist(x=transX, y=y, committees=25)
	summary(finalModel)

	# transform the validation dataset
	set.seed(7)
	valX <- validation[,1:13]
	trans_valX <- predict(preprocessParams, valX)
	valY <- validation[,14]
	# use final model to make predictions on the validation dataset 
	predictions <- predict(finalModel, newdata=trans_valX, neighbors=3) # calculate RMSE
	rmse <- RMSE(predictions, valY)
	r2 <- R2(predictions, valY)
	print(rmse)
}
finalize_model(dataset)