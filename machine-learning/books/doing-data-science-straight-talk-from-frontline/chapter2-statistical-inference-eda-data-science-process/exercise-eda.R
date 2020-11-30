library(ggplot2)

PATH <- "./dds_ch2_nyt/nyt1.csv"

ds <- read.csv(PATH, header=TRUE)
# List first five rows
# head(ds)

# List last five rows
# tail(ds)

# Show number if rows and columns
# dim(ds) 

# Show column names
# names(ds)

# Show column names with data types
# sapply(ds, class) 

# Generate Descriptive Statistics
# summary(ds)

#Create a new variable, age_group, that categorizes users as "<18", "18-24", "25-34", "35-44", "45-54", "55-64", and "65+".
# Filter by age
age_group1 <- ds[ds$Age < 18, ]
age_group2 <- ds[ds$Age >= 18 & ds$Age <= 24, ]
age_group3 <- ds[ds$Age >= 25 & ds$Age <= 34, ]
age_group4 <- ds[ds$Age >= 35 & ds$Age <= 44, ]
age_group5 <- ds[ds$Age >= 45 & ds$Age <= 54, ]
age_group6 <- ds[ds$Age >= 55 & ds$Age <= 64, ]
age_group7 <- ds[ds$Age >= 65, ]

#NOTE Replace number to plot each age category my example is age_group7

# Plot the distributions of number impressions and click- through-rate (CTR=# clicks/# impressions) for these six age categories.
# d_impressions <- density(age_group7 $Impression)
# pdf(file="myplot.pdf")
# plot(d_impressions, main="Number of Impression")
# polygon(d_impressions, col="blue", border="blue")
# dev.off()
# browseURL("myplot.pdf")

# pdf(file="myplot.pdf")
# x <- age_group7$Clicks
# h <- hist(x, breaks=10, col="red", xlab="Number of Clicks", main="Histogram with Curve")
# xfit <- seq(min(x), max(x), length=40) 
# yfit <- dnorm(xfit, mean=mean(x), sd=sd(x)) 
# yfit <- yfit * diff(h$mids[1:2]) * length(x) 
# lines(xfit, yfit, col="blue", lwd=2)
# dev.off()
# browseURL("myplot.pdf")

# Define a new variable to segment or categorize users based on their click behavior.
age_group7_with_clicks <- age_group7[age_group7$Clicks > 0, ]
# age_group7_without_clicks <- age_group7[age_group7$Clicks <= 0, ]

# Explore the data and make visual and quantitative comparisons across user segments/demographics (<18-year-old males ver‐ sus < 18-year-old females or logged-in versus not, for example).
# head(age_group7_with_clicks)
m <- age_group7_with_clicks[age_group7_with_clicks$Gender == 0 & age_group7_with_clicks < 18, ]
f <- age_group7_with_clicks[age_group7_with_clicks$Gender == 1 & age_group7_with_clicks < 18, ]