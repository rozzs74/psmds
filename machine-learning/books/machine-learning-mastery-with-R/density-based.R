library(mlbench)
library(dbscan)

data(PimaIndiansDiabetes)

ds <- PimaIndiansDiabetes

dim(ds)
names(ds)
sapply(ds, class)


eps <- 100
min_pts <- 200

gen_seed <- function(n) {
    set.seed(n)
    return(TRUE)
}

gen_seed(7)
m <- ds[,1:8]
pmd_as_matrix <- as.matrix(m)

# dbscan
# db <- dbscan(pmd_as_matrix, eps, min_pts)
# hullplot(pmd_as_matrix, db$cluster)

# # hdbscan
hdb <- hdbscan(pmd_as_matrix, minPts =4)
hdb

colors <- mapply(function(col, i) adjustcolor(col, alpha.f = hdb$membership_prob[i]), 
palette()[hdb$cluster+1], seq_along(hdb$cluster))
plot(pmd_as_matrix, col=colors, pch=20)