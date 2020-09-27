
#############################################
# Author: John Royce C. Punay               #
# Date created: September 27, 2020, 9:25PM  #
#############################################

library(arules)


ASSOC_RULES <- "rules"
MOST_FREQUENT <- "frequent itemsets"

assoc_rules <- function(type, df) {
	rules_params <- list(support=0.01, confidence=0.05, target=type)
	rules_params
	assoc.rules <- apriori(df, parameter=rules_params)
	assoc.rules
	summary(assoc.rules)
	inspect(head(sort(assoc.rules, by = "confidence"), 10))
	inspect(head(sort(assoc.rules,by="support"),10))
	return(TRUE)
}

most_frequnt <- function(type, df) {
	most_frequent_params <- list(support=0.01, confidence=0.05, target=type)
	most_frequent_params
	most.frequent <- apriori(df, parameter=most_frequent_params)
	most.frequent
	summary(most.frequent)
	inspect(head(sort(most.frequent, by = "support"), 10))
}


#Data Preprocessing
transactions <-  c(1,2,3,4,5,6,7,8,9,10)
items <- c("I1, I2, I3, I4", "I1, I3, I5, I6", "I7, I1, I2", "I5, I8, I7, I4", "I5, I3, I4, I1", "I8, I1, I5, I6", "I1, I2, I5, I3", "I2, I3, I4, I1", "I3, I1, I8, I4", "I4, I8, I7, I5")
transactions
items
df <- data.frame(transactions, items)
df

#Apriori
gen_assoc_rules <- assoc_rules(ASSOC_RULES, df)
gen_most_frequent <- most_frequnt(MOST_FREQUENT, df)