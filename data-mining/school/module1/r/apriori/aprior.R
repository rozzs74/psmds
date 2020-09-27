library(arules)
data(Groceries)

summary(Groceries) 
class(Groceries)
Groceries@itemInfo[1,] # First item
Groceries@itemInfo[1:20,] # Twnety rows of sparse items

#The following code displays the lOth to 20th transactions of the Groceries dataset.
apply(Groceries@data[,10:20],2,function(r)    paste(Groceries@itemInfo[r,"labels"],collapse=", "))

# Apriori algo
# First, get itemsets of length 1
params <- list(minlen=1, maxlen=1, support=0.02, target="frequent itemsets")
result <- apriori(Groceries, parameter=params)
result
summary(result)
inspect(head(sort(result,by="support"),10))   # lists top 10