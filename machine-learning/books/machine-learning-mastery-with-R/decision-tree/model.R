#############################################
# Author: John Royce C. Punay               #
# Date created: September 24, 2020, 7:15PM  #
#############################################


library(mlbench)
library(rpart)
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

model.dt <- rpart(diabetes~., data=trainData)
summary(model.dt)
print(model.dt)

predictions <- predict(model.dt, testData[,1:8], type="class")
summary(predictions)
head(predictions)
table(predictions, testData$diabetes)
plot(predictions)
confusionMatrix(predictions ,testData$diabetes)