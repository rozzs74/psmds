#############################################
# Author: John Royce C. Punay               #
# Date created: September 19, 2020, 8:13PM  #
#############################################

library(caret)
path <- "/Users/cbo/Desktop/psmds/data-mining/school/module1/r/polynomial-regression/who_suicide_statistics.csv"
data_sets <- read.csv(path, header=TRUE)
head(data_sets)
tail(data_sets)
# datasets column name
names(data_sets)

#utils
count <- function(ds) {
    dim(ds)
}

# dimension to get number of rows and columns
count(data_sets)

# data type
sapply(data_sets, class)

# data summary for statistics
summary(data_sets) 

# utils
empty <- function(v) {
    sum(is.na(v))
}
fill <- function(x) { 
    x[is.na(x)] <- 0; x
}

data_sets$suicides_no =  fill(data_sets$suicides_no)
data_sets$population =  fill(data_sets$population)
empty(data_sets)
tail(data_sets)

unique(data_sets[,1:6])

count(data_sets)
year_y <- data_sets$year
population_x <- data_sets$population

hist(year_y, 
     main="Histogram for yearly suicidal deaths", 
     xlab="Year", 
     border="red", 
     col="blue",
     las=1, 
     breaks=5)


hist(population_x, 
     main="Histogram base on population", 
     xlab="Population", 
     border="green", 
     col="pink",
     las=1, 
     breaks=5)

set.seed(100)
trainControl <- trainControl(method="cv", number=5)
trainIndex <- createDataPartition(year_y, p=0.80, list=FALSE)
dataTrain <- data_sets[ trainIndex,]
dataTest <- data_sets[-trainIndex,]

model <- lm(year_y~population_x, data=dataTest)
summary(model)

predictions <- predict(model, dataTest)
print(predictions)


RMSE(predictions, data_sets$year)
R2(predictions, data_sets$year)


a <- 0.5
p <- seq(0,100,1)
y <- p*a
# plot(p,y,type='l',col='red',main='Linear relationship')
y <- 450 + a*(p-10)^3
plot(p,y,type='l',col='navy',main='Polynomial relationship',lwd=3)