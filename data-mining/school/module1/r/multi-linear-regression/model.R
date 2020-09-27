library(caret)
library(klaR)

# path
path <- "../data-sets.csv"
dataset <- read.csv(path, header=TRUE)

# Convert categorical to numerical
dataset$Country.or.region <- as.numeric(factor(dataset$Country.or.region))
class(dataset$Country.or.region)

# Descriptive statistics and peeking data
head(dataset)
names(dataset)
dim(dataset)
summary(dataset)

# Data preprocessing
sum(is.na(dataset))
missingdata <- dataset[!complete.cases(dataset)]
sum(is.na(missingdata))

# resampling 80/20
set.seed(100)
trainControl <- trainControl(method="cv", number=5)
trainIndex <- createDataPartition(dataset$Score, p=0.80, list=FALSE)
dataTrain <- dataset[ trainIndex,]
dataTest <- dataset[-trainIndex,]

# training
model <- lm(Score ~., data=dataset)
summary(model)

#predictions
predictions <- predict(model, dataTest)
print(predictions)

# Evaluate
RMSE(predictions, dataTest$Score)
R2(predictions, dataTest$Score)
