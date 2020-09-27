# How to load your data from a CSV file located on a webserver.
library(RCurl)

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
output <- getURL(url, ssl.verifypeer=FALSE)
stream <- textConnection(output)
dataset <- read.csv(stream, header=FALSE)
head(dataset)
