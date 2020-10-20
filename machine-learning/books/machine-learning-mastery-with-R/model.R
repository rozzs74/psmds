library(caret)
library(klaR)
path <- "./data-sets.csv"
dataset <- read.csv(path, header=TRUE)
dataset$Country.or.region <- as.numeric(factor(dataset$Country.or.region))
class(dataset$Country.or.region)
head(dataset)
names(dataset)
dim(dataset)
summary(dataset)
sum(is.na(dataset))
missingdata <- dataset[!complete.cases(dataset)]
sum(is.na(missingdata))

set.seed(100)
trainControl <- trainControl(method="cv", number=5)
# #80/20 resampling     
trainIndex <- createDataPartition(dataset$Score, p=0.80, list=FALSE)
dataTrain <- dataset[ trainIndex,]
dataTest <- dataset[-trainIndex,]

model <- lm(Score ~., data=dataTest)
summary(model)
predictions <- predict(model, dataTest)
print(predictions)

RMSE(predictions, dataTest$Score)
R2(predictions, dataTest$Score)

