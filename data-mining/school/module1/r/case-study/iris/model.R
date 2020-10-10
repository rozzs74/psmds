library(caret)

data(iris)
dataset <- iris


validationIndex <- createDataPartition(dataset$Species, p=0.80, list=FALSE)

# 20% for validation
validation <- dataset[-validationIndex, ]

# 80% for training and testing
dataset <- dataset[validationIndex,]


# # datasets dimensions
# dim(dataset)
# sapply(dataset, class)


# # Peek at the data
# head(dataset)

# # Levels of the Class
# levels(dataset$Species)


# Class distribution
# percentage <- prop.table(table(dataset$Species)) * 100
# cbind(freq=table(dataset$Species), percentage=percentage)

summary(dataset)