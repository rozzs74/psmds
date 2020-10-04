library(mlbench)
library(dbscan)
library(ggplot2)

data(PimaIndiansDiabetes)

ds <- PimaIndiansDiabetes

dim(ds)
names(ds)
sapply(ds, class)

eps <- 20
min_pts <- 6

gen_seed <- function(n) {
    set.seed(n)
    return(TRUE)
}

gen_seed(7)
m <- ds[,1:8]
pmd_as_matrix <- as.matrix(m)

# dbscan
db <- dbscan(pmd_as_matrix, eps, min_pts)
db$cluster
hullplot(pmd_as_matrix, db$cluster, main="DBSCAN")