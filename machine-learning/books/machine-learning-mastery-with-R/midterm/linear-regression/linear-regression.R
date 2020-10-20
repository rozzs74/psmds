#############################################
# Author: John Royce C. Punay               #
# Date created: September 27, 2020, 9:11 AM #
#############################################

library(caret)
library(klaR)

data_frames <- "../poverty-datasets-raw.csv"

read_csv <- function(path, show_header) {
    val <- read.csv(path, header = TRUE)
    return(val)
}

write_csv <- function(data_frames,path, show_column_names) {
    write.csv(data_frames, path, row.names = show_column_names)
}

gen <- function(s, min, max) {
    f <- runif(s, min=min, max=max)
    return(f)
}

plot_histogram <- function(data_frames) {
    par(mfrow=c(1,6))
    for(i in 3:8) {
        hist(data_frames[,i], main=names(data_frames)[i])
    } 
}

plot_density <- function(data_frames) {
    par(mfrow=c(1,6))
    for(i in 3:8) {
        plot(density(data_frames[,i]), main=names(data_frames)[i])
    } 
}

plot_box_and_whisker <- function(data_frames) {
    par(mfrow=c(1,6))
    for(i in 3:8) {
        boxplot(data_frames[,i], main=names(data_frames)[i])
    }  
}

#UNDERSTANDING YOUR DATA
raw_data_frames <- read_csv(data_frames_path, TRUE)


# PREPROCESSING
country <- unique(raw_data_frames$Country.Name)
country_code <- unique(raw_data_frames$Country.Code)
year <- unique(2009)
population <- unique(raw_data_frames$X2009..YR2009)
annualized_growth_income <- sample(length(country))
electricity_poverty_deprived <- gen(length(country), 0, 40)
water_poverty_deprived <- gen(length(country), 0, 40)
education_poverty_deprived <- gen(length(country), 0, 40)
data_frames <- data.frame(country, country_code, year, population, annualized_growth_income, electricity_poverty_deprived, water_poverty_deprived, education_poverty_deprived)
data_frames <- data_frames[!(is.na(data_frames$country) | data_frames$country==""), ]
head(data_frames)
tail(data_frames)
names(data_frames)
dim(data_frames)
sapply(data_frames, class)
attr <- data_frames[,3:8]
summary(attr)
preprocessParams <- preProcess(attr, method=c("center", "scale"))
print(preprocessParams)
Y <- data_frames$population
set.seed(7)
trainControl <- trainControl(method="cv", number=5)
trainIndex <- createDataPartition(Y, p=0.80, list=FALSE)
trainData <- data_frames[trainIndex,]
testData <- data_frames[-trainIndex,]
head(trainData)
head(testData)
head(Y)
write_csv(data_frames,"../poverty-cleaned.csv", TRUE)


# Visualization
plot_histogram(data_frames)
plot_density(data_frames)
plot_box_and_whisker(data_frames)


#Modeling
model.lr <- lm(data_frames$population ~ data_frames$annualized_growth_income, data=trainData) 
summary(model.lr)


predictions <- predict(model.lr, testData)
print(predictions)
summary(predictions)

table(predictions, trainData$population)

RMSE(predictions, trainData$population)
R2(predictions, trainData$population)

anova(model.lr, test="Chisq")