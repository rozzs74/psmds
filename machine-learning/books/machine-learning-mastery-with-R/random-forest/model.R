#############################################
# Author: John Royce C. Punay               #
# Date created: September 24, 2020, 9:08 PM  #
#############################################

library(randomForest)
library(mlbench)
library(caret)

data(PimaIndiansDiabetes)

#data sets dimention
str(PimaIndiansDiabetes)
head(PimaIndiansDiabetes)
tail(PimaIndiansDiabetes)
names(PimaIndiansDiabetes)
dim(PimaIndiansDiabetes)
sapply(PimaIndiansDiabetes, class)
attr <- PimaIndiansDiabetes[,1:8]
summary(attr)


#preprocess
preprocessParams <- preProcess(attr, method=c("center", "scale"))
print(preprocessParams)

#resampling
Y <- PimaIndiansDiabetes$diabetes
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
trainIndex <- createDataPartition(Y, p=0.80, list=FALSE)
trainData <- PimaIndiansDiabetes[trainIndex,]
testData <- PimaIndiansDiabetes[-trainIndex,]

head(trainData)
head(testData)
head(Y)

model.rf <- randomForest(diabetes ~., data=trainData, mtry=8, ntree=50, importance=TRUE)
summary(model.rf)
print(model.rf)

#predictions
predictions <- predict(model.rf, testData[,1:8])
summary(predictions)
head(predictions)
table(predictions, testData$diabetes)

#model eval
confusionMatrix(predictions ,testData$diabetes)

par(mfrow = c(1, 2))
varImpPlot(model.rf, type = 2, main = "Variable Importance",col = 'black')
plot(model.rf, main = "Error vs no. of trees grown")
