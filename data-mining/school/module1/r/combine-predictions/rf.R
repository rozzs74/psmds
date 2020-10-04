# load packages
library(caret) 
library(mlbench) 
library(randomForest) 



# load dataset
data(Sonar) 
set.seed(7)

# create 80%/20% for training and validation datasets
validationIndex <- createDataPartition(Sonar$Class, p=0.80, list=FALSE) 
validation <- Sonar[-validationIndex,]
training <- Sonar[validationIndex,]


# train a model and summarize model
set.seed(7)
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)


fit.rf <- train(Class~., data=training, method="rf", metric="Accuracy", trControl=trainControl, ntree=2000)
# print(fit.rf)

# print(fit.rf$finalModel)
# create standalone model using all training data
set.seed(7)
finalModel <- randomForest(Class~., training, mtry=2, ntree=2000)
# make a predictions on "new data" using the final model 
finalPredictions <- predict(finalModel, validation[,1:60])
confusionMatrix(finalPredictions, validation$Class)