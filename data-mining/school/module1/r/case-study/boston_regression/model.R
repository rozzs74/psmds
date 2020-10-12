library(mlbench)
library(caret)
library(corrplot)

data(BostonHousing)

# Split out validation dataset
# create a list of 80% of the rows in the original dataset we can use for training
set.seed(7)
validationIndex <- createDataPartition(BostonHousing$medv, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- BostonHousing[-validationIndex,]
# use the remaining 80% of data to training and testing the models
dataset <- BostonHousing[validationIndex,]

dataset$chas <- as.numeric(as.factor(dataset$chas))

dim(dataset)
sapply(dataset, class)

summary(dataset)
cor(dataset[,1:13])

pdf(file="test.pdf")
quartz("Test")
par(mfrow=c(2,7))
for(i in 1:13) {
  hist(dataset[,i], main=names(dataset)[i])
}
dev.off()
savePlot()