library(mlbench)
library(caret)
library(corrplot)

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
# correlations <- cor(dataset[,1:13]) 
# corrplot(correlations, method="circle")



run_linear_algorithms <- function(dataset, metric) {
	#LM
	set.seed(7)
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	fit.lm <- train(medv~., data=dataset, method="lm", metric=metric, preProc=c("center", "scale"), trControl=trainControl)
	#GLM
	set.seed(7)
	fit.glm <- train(medv~., data=dataset, method="glm", metric=metric, preProc=c("center", "scale"), trControl=trainControl)
	#GLMNET
	set.seed(7)
	fit.glmnet <- train(medv~., data=dataset, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=trainControl)
	return(list(lm=fit.lm, glm=fit.glm, glmnet=fit.glmnet))
}

run_non_linear_algorithms <- function(dataset, metric) {
	#SVM
	set.seed(7)
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=trainControl)

	#CART
	set.seed(7)
	grid <- expand.grid(.cp=c(0, 0.05, 0.1))
	fit.cart <- train(medv~., data=dataset, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale"), trControl=trainControl)

	#KNN
	set.seed(7)
	fit.knn <- train(medv~., data=dataset, method="knn", metric=metric, preProc=c("center", "scale"), trControl=trainControl)
	return(list(svm=fit.svm, cart=fit.cart, knn=fit.knn))
}

linears <- run_linear_algorithms(dataset, "RMSE")
non_linears <- run_non_linear_algorithms(dataset, "RMSE")
results <- resamples(list(LM=linears$lm, GLM=linears$glm, GLMNET=linears$glmnet, SVM=non_linears$svm, CART=non_linears$cart, KNN=non_linears$knn))
summary(results)
dotplot(results)
