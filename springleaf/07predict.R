library("xgboost")
source("utils.R")

load("test06.RData")
dtrain = xgb.DMatrix("train06.DMatrix")

features = get_features(testdf)
watchlist = list(train = dtrain)
params = list(objective   = "binary:logistic", 
              eta         = 0.02,
              max.depth   = 16,
              subsample   = 0.7,
              eval_metric = "auc")

clf = xgb.train(params              = params,
                data                = dtrain, 
                nrounds             = 5000, 
                watchlist           = watchlist,
                print.every.n       = 10)

rm(dtrain)
gc()
submission = data.frame(id=testdf$id)
submission$target = NA 
submission[,"target"] = predict(clf, data.matrix(testdf[,features]))
write.csv(submission, "xgb_sub2.csv", row.names=F)

