#############################################
# Author: John Royce C. Punay               #
# Date created: September 23, 2020, 6:29PM  #
#############################################

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

#modeling
model.knn <- knn3(diabetes~., data=trainData, k=3)
summary(model.knn)
print(model.knn)

#prediction
predictions <- predict(model.knn, testData[,1:8], type="class")
summary(predictions)
head(predictions)
table(predictions, testData$diabetes)

confusionMatrix(predictions ,testData$diabetes)