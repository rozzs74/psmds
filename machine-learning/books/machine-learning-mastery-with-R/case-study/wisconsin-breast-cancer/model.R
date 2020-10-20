library(caret)
library(mlbench)

data(BreastCancer)

set.seed(7)
validationIndex <- createDataPartition(BreastCancer$Class, p=0.80, list=FALSE)

validation <- validationIndex[-validationIndex, ] #test dataset 20
dataset <- BreastCancer[validationIndex, ] #training dataset 80

# dim(dataset)
# head(dataset, n=20)

dataset <- dataset[, -1]
# convert input values to numeric
for(i in 1:9) {
    dataset[, i] <- as.numeric(as.character(dataset[,i]))
}

# sapply(dataset, class)
# tail(dataset)
dataset[is.na(dataset)] <- 0
# sum(is.na(dataset))

# complete_cases <- complete.cases(dataset) 
# cor(dataset[complete_cases,1:9])
# percentage <- prop.table(table(dataset$Class))*100
# cbind(freq=table(dataset$Class), percentage=percentage)
# levels(dataset$Class)

unimodal_visualizations <- function(dataset) {
    par(mfrow=c(3,3))
    for(i in 1:9) {
        hist(dataset[,i], main=names(dataset)[i])
    }
    par(mfrow=c(3, 3))
    complete_cases <- complete.cases(dataset)
    for(i in 1:9) {
        plot(density(dataset[complete_cases,i]), main=names(dataset)[i])
    }
    par(mfrow=c(3, 3))
    for(i in 1:9) {
        boxplot(dataset[,i], main=names(dataset)[i])
    }
    
}

multi_modal_visualizations <- function(dataset) {
    # scatterplot matrix
    jittered_x <- sapply(dataset[,1:9], jitter)
    pairs(jittered_x, names(dataset[,1:9]), col=dataset$Class)

    par(mfrow=c(3, 3))
    for(i in 1:9) {
        barplot(table(dataset$Class, dataset[, i]), main=names(dataset)[i], legend.text=unique(dataset$Class))
    }
}

run_linear_algorithms <- function(dataset, metric, preProc) {
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	# #LG
    set.seed(7)
	model.fit.lg <- train(Class~., data=dataset, method="glm", metric=metric, trControl=trainControl, preProc=preProc)

	#LDA
	set.seed(7)
	model.fit.lda <- train(Class ~ ., data=dataset, method="lda", metric=metric, trControl=trainControl, preProc=preProc)

	#GLMNET
	set.seed(7)
	model.fit.glmnet <- train(Class ~ ., data=dataset, method="glmnet", metric=metric, trControl=trainControl, preProc=preProc)

	return(list(lg=model.fit.lg, lda=model.fit.lda, glm=model.fit.glmnet))
}

run_non_linear_algorithms <- function(dataset, metric, preProc) {
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)

	#KNN
	set.seed(7)
	model.fit.knn <- train(Class~., data=dataset, method="knn", metric=metric, trControl=trainControl, preProc=preProc)

	#CART
	set.seed(7)
	model.fit.cart <- train(Class~., data=dataset, method="rpart", metric=metric, trControl=trainControl, preProc=preProc)

	#Naive Bayes
	set.seed(7)
	model.fit.nb <- train(Class~., data=dataset, method="nb", metric=metric, trControl=trainControl, preProc=preProc)

	#SVM
	set.seed(7)
	model.fit.svm <- train(Class~., data=dataset, method="svmRadial", metric=metric, trControl=trainControl, preProc=preProc)

	return(list(knn=model.fit.knn, cart=model.fit.cart, nb=model.fit.nb, svm=model.fit.svm))
}

# linears <- run_linear_algorithms(dataset, "Accuracy", c("BoxCox"))
# non_linears <- run_non_linear_algorithms(dataset, "Accuracy", c("BoxCox"))

# results <- resamples(list(LG=linears$lg, LDA=linears$lda, GLM=linears$glm, KNN=non_linears$knn, CART=non_linears$cart, NB=non_linears$nb, SVM=non_linears$svm))
# summary(results)
# dotplot(results)

tune_svm <- function(dataset, metric, preProc) {
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	set.seed(7)
	grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
	model.fit.svm <- train(Class ~ ., data=dataset, method="svmRadial", metric=metric, trControl=trainControl, preProc=preProc, tuneGrid=grid)
	print(model.fit.svm)
	plot(model.fit.svm)
}


tune_knn <- function(dataset, metric, preProc) {
	trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
	set.seed(7)
	grid <- expand.grid(.k=seq(1, 20, by=1))
	model.fit.knn <- train(Class ~ ., data=dataset, method="knn", metric=metric, trControl=trainControl, preProc=preProc, tuneGrid=grid)
	print(model.fit.knn)
	plot(model.fit.knn)
}


final_model <- function(dataset) {
	# prepare parameters for data transform
	set.seed(7)
	datasetNoMissing <- dataset[complete.cases(dataset),]
	x <- datasetNoMissing[,1:9]
	preprocessParams <- preProcess(x, method=c("BoxCox"))
	x <- predict(preprocessParams, x)

	# prepare the validation dataset
	set.seed(7)
	# remove id column
	validation <- dataset[, -1]	
	# remove missing values (not allowed in this implementation of knn)
	validation <- validation[complete.cases(validation),]
	# convert to numeric
	for(i in 1:9) {
		validation[,i] <- as.numeric(as.character(validation[,i])) 
	}
}
# 	# transform the validation dataset
# 	validationX <- predict(preprocessParams, validation[,1:9])


# 	# make predictions
# 	set.seed(7)
# 	predictions <- knn3Train(x, validationX, datasetNoMissing$Class, k=9, prob=FALSE) 
# 	confusionMatrix(predictions, validation$Class)
# }

final_model(ds)
