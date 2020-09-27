#############################################
# Author: John Royce C. Punay               #
# Date created: September 23, 2020, 7:49PM  #
#############################################

library(kernlab)
library(mlbench)
library(caret)
data(PimaIndiansDiabetes)

#data sets dimension
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

#modeling
model.svm <- ksvm(diabetes~., data=trainData, kernel="rbfdot")
summary(model.svm)
print(model.svm)

predictions <- predict(model.svm, testData[,1:8], type="response")
summary(predictions)
head(predictions)
predictions
table(predictions, testData$diabetes)

confusionMatrix(predictions ,testData$diabetes)