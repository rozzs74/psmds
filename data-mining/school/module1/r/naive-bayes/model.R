#############################################
# Author: John Royce C. Punay               #
# Date created: September 24, 2020, 8:26 PM  #
#############################################


library(e1071)
library(caret)
library(mlbench)
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

model.nb <- naiveBayes(diabetes~., data=trainData)
summary(model.nb)
print(model.nb)

predictions <- predict(model.nb, testData[,1:8])
summary(predictions)
head(predictions)
table(predictions, testData$diabetes)
confusionMatrix(predictions ,testData$diabetes)