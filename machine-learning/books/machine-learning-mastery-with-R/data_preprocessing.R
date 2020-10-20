library(readr)
library(plotrix)
path <- "./Cleaned-Data.csv" 

data_frames <- read_csv(path)

names(data_frames)
spec(data_frames)
tail(data_frames)

sum(is.na(data_frames))

sample(data_frames)

missingdata <- data_frames[!complete.cases(data_frames), ]
# missingdata

sum(is.na(missingdata))

# slices <- c(data_frames$Pains)
# lbls <- c(data_frames$Country)
# pie(slices, labels = lbls, main="Pie Chart of COVID-19 with fever")
# pains <- table(data_frames$Pains,)
# barplot(pains, main="Suffering", xlab="No of Pains")
# counts <- table(data_frames$Pains, data_frames$Country)
# barplot(counts, main="No of Pained over country",xlab="Country", col=c("darkblue","red"),legend = rownames(counts), beside=TRUE)
# slices <- c(data_frames$Tiredness)
# lbls <- c(data_frames$Country)
# pie3D(data_frames$Tiredness,data_frames$Country=lbls,explode=0.1,main="Pie Chart of Countries")


# slices <- c(data_frames$Number_High_Severity)
# lbls <- c(data_frames$Country)
# pie3D(slices,labels=lbls,explode=0.1, main="Number of high severity per country")
# https://stackoverflow.com/questions/49138895/convert-multiple-categorical-variables-to-factors-in-r
# http://www.sthda.com/english/articles/40-regression-analysis/165-linear-regression-essentials-in-r/
# http://r-statistics.co/Linear-Regression.html