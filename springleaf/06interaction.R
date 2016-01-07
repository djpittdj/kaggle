# following Owen Zhang's Amazon solution
library(xgboost)
source("utils.R")

load('train05.RData')
load('test05.RData')
testdf[,"target"] = NA
mergedf = rbind(traindf, testdf)
nrow_train = nrow(traindf)
nrow_test = nrow(testdf)
nrow_total = nrow_train + nrow_test

imp_features = read.table("combined_features.csv", header=F)
for (feature in imp_features[,"V1"]) {
  mergedf[, paste0(feature,"_c")] = count1way(mergedf, feature)
}

traindf = mergedf[1:nrow_train,]
testdf = mergedf[(nrow_train+1):nrow_total,]
save(traindf, file="train06.RData")
save(testdf, file="test06.RData")
write.csv(traindf, file="train06.csv", row.names=F)
write.csv(testdf, file="test06.csv", row.names=F)

features = get_features(traindf)
dtrain = xgb.DMatrix(data.matrix(traindf[, features]), label=traindf$target)
xgb.DMatrix.save(dtrain, "train06.DMatrix")

write.table(features, file="features06.csv", row.names=F, col.names=F, quote=F)
