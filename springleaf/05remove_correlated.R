library(caret)
source("utils.R")

load("train04.RData")
load("test04.RData")
load("cor_mat.RData")

features = get_features(traindf)

cor_cols = findCorrelation(cor_mat, cutoff=0.99)
cor_cols_names = features[cor_cols]

traindf = del_cols(traindf, cor_cols_names)
testdf = del_cols(testdf, cor_cols_names)

save(traindf, file="train05.RData")
save(testdf, file="test05.RData")
write.csv(traindf, file="train05.csv", row.names=F)
write.csv(testdf, file="test05.csv", row.names=F)
