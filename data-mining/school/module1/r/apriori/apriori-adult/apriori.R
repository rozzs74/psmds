library(arules)
data(Adult)

params <- list(supp=0.5, conf=0.5, target="rules")
rules <- apriori(Adult, parameter=params)
summary(rules)

inspect(rules) #It gives the list of all significant association rules. Some of them are shown below