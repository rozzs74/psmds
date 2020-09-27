#############################################
# Author: John Royce C. Punay               #
# Date created: September 22, 2020, 8:13PM  #
#############################################

library(mlbench)
library(caret)

data(PimaIndiansDiabetes)

#data sets dimention
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
model.lr <- glm(diabetes~., data=trainData, family=binomial(link='logit')) 
summary(model.lr)

#prediction
probabilities <- predict(model.lr, data=testData, type='response')
predictions <- ifelse(probabilities > 0.5,'pos','neg')
head(predictions)
table(predictions, trainData$diabetes)


# model evaluation
anova(model.lr, test="Chisq")

model.lrg <- glm(diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass + pedigree + age, data=trainData,family = binomial(link="logit"))
summary(model.lrg)

# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/logistic-regression-analysis-r/tutorial/