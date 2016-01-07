
traindf = read.csv("../train.csv")
testdf  = read.csv("../test.csv")

names(traindf) = gsub("_", "", tolower(names(traindf)))
names(testdf) = gsub("_", "", tolower(names(testdf)))

save(traindf, file="train01.RData")
save(testdf, file="test01.RData")
