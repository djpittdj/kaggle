source("bigCorPar.R")
source("utils.R")

load("train04.RData")
features = get_features(traindf)
n_feat = length(features)

cor_mat = matrix(NA, nrow=n_feat, ncol=n_feat)
m = bigCorPar(traindf[,features], nblocks=17, ncore=4)
for (i in 1:n_feat) {
  if (i%%10==0) {
    cat(i,"\n")
  }
  for (j in 1:n_feat) {
    cor_mat[i,j] = m[i,j][[1]]
  }
}

save(cor_mat, file="cor_mat.RData")
