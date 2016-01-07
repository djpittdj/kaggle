source("utils.R")

load("train01.RData")
load("test01.RData")

# count number of missing values in each row
traindf["num_na"] = NA
testdf["num_na"] = NA
for (rows in split(1:nrow(traindf), ceiling((1:nrow(traindf))/10000))) {
  traindf[rows, "num_na"] = apply(traindf[rows, ], 1, function(x) sum(is.na(x)))
}

for (rows in split(1:nrow(testdf), ceiling((1:nrow(testdf))/10000))) {
  testdf[rows, "num_na"] = apply(testdf[rows, ], 1, function(x) sum(is.na(x)))
}

gc()

# fill in missing values and -99999
traindf[is.na(traindf)] = -2
testdf[is.na(testdf)] = -2
traindf[traindf==-99999] = -3
testdf[testdf==-99999] = -3
gc()

traindf_nrow = nrow(traindf)
testdf_nrow = nrow(testdf)
total_nrow = traindf_nrow + testdf_nrow

# delete these factor columns that has only one unique element and missing values
cols_to_remove = c("var0008", "var0009", "var0010", "var0011", "var0012", "var0043", "var0044", "var0196", "var0202", "var0216", "var0222", "var0229", "var0239")
traindf = del_cols(traindf, cols_to_remove)
testdf = del_cols(testdf, cols_to_remove)
gc()

# time difference between var0073 and var0075 in terms of days
feature1 = "var0073"
feature2 = "var0075"
traindf["diff73_75"] = as.numeric(as.Date(substr(traindf[, feature1], 1, 7), format="%d%b%y") - as.Date(substr(traindf[, feature2], 1, 7), format="%d%b%y"))
testdf["diff73_75"] = as.numeric(as.Date(substr(testdf[, feature1], 1, 7), format="%d%b%y") - as.Date(substr(testdf[, feature2], 1, 7), format="%d%b%y"))
traindf["diff73_75"][is.na(traindf["diff73_75"])] = -1
testdf["diff73_75"][is.na(testdf["diff73_75"])] = -1

# time difference between var0204 and var0217 in terms of days
feature1 = "var0204"
feature2 = "var0217"
traindf["diff204_217"] = as.numeric(as.Date(substr(traindf[, feature1], 1, 7), format="%d%b%y") - as.Date(substr(traindf[, feature2], 1, 7), format="%d%b%y"))
testdf["diff204_217"] = as.numeric(as.Date(substr(testdf[, feature1], 1, 7), format="%d%b%y") - as.Date(substr(testdf[, feature2], 1, 7), format="%d%b%y"))
traindf["diff204_217"][is.na(traindf["diff204_217"])] = -1
testdf["diff204_217"][is.na(testdf["diff204_217"])] = -1

# factorize time columns. Only use the month and year info
cols_time = c("var0073","var0075","var0156","var0157","var0158","var0159","var0166","var0167","var0168","var0169","var0176","var0177","var0178","var0179","var0204","var0217")
for (feature in cols_time) {
  l1 = substr(traindf[,feature], 1, 7)
  l2 = substr(testdf[,feature], 1, 7)
  uniq_levels = unique(c(l1, l2))
  traindf[,feature] = as.integer(factor(l1, levels=uniq_levels))
  testdf[,feature] = as.integer(factor(l2, levels=uniq_levels))
}

# factorize time column var531
feature = "var0531"
uniq_levels = unique(c(traindf[,feature], testdf[,feature]))
traindf[,feature] = as.integer(factor(traindf[,feature], levels=uniq_levels))
testdf[,feature] = as.integer(factor(testdf[,feature], levels=uniq_levels))

# deal with the zipcode and the city
var0212 = c(traindf[,"var0212"], testdf[,"var0212"])
var0212 = substr(var0212, 1, 5)
var0241 = c(traindf[,"var0241"], testdf[,"var0241"])
var0212[is.na(var0212)] = var0241[is.na(var0212)]
var0212[is.na(var0212)] = -1
uniq_levels = unique(var0212)
var0212 = as.integer(factor(var0212, levels=uniq_levels))
traindf[,"var0212"] = var0212[1:traindf_nrow]
testdf[,"var0212"] = var0212[(traindf_nrow+1):total_nrow]
cols_zipcode = c("var0241", "var0200")
traindf = del_cols(traindf, cols_zipcode)
testdf = del_cols(testdf, cols_zipcode)

# replace occupation columns var0404 and var0493 with processed data
for (var_num in c("var0404", "var0493")) {
  var = read.table(paste0(var_num, "_proc.dat"), header=F)
  traindf[, var_num] = var[1:traindf_nrow, "V1"]
  testdf[, var_num] = var[(traindf_nrow+1):total_nrow, "V1"]
}

# cut numeric features into cateogories
cols_numeric = c("var0299","var0300","var0301","var0319","var0320","var0321","var0335","var0336","var0337","var0370")
for (feature in cols_numeric) {
  var = data.frame(c(traindf[,feature], testdf[,feature]))
  names(var) = "val"
  var[var>0] = as.character(cut(var[var>0], quantile(var[var>0], probs=seq(0,1,by=0.2)), labels=c("A","B","C","D","E"), include.lowest=T))
  var[var==-1] = "X"
  var[var==0] = "Y"
  var = as.integer(factor(var$val))
  traindf[,feature] = var[1:traindf_nrow]
  testdf[,feature] = var[(traindf_nrow+1):total_nrow]
}

# factorize other factor columns
features = get_features(traindf)

for (feature in features) {
  if (class(traindf[[feature]])=="factor") {
    uniq_levels = unique(c(as.character(traindf[[feature]]), as.character(testdf[[feature]])))
    traindf[[feature]] = as.integer(factor(traindf[[feature]], levels=uniq_levels))
    testdf[[feature]]  = as.integer(factor(testdf[[feature]],  levels=uniq_levels))
  }
}

# remove the columns with one unique value
features = get_features(traindf)

cols_one_val = vector()
for (feature in features) {
  if (length(unique(traindf[,feature])) == 1) {
    cols_one_val = c(cols_one_val, feature)
    cat(sprintf("removing feature %s with only one unique value\n", feature))
  }
}

traindf = del_cols(traindf, cols_one_val)
testdf = del_cols(testdf, cols_one_val)

# remove features that are all the same
features = get_features(traindf)
num_features = length(features)

identical_cols_list = list()
for (i in 1:(num_features-1)) {
  feature_i = features[i]
  identical_cols = vector()
  for (j in (i+1):num_features) {
    feature_j = features[j]
    if (identical(traindf[[feature_i]], traindf[[feature_j]])) {
      identical_cols = c(identical_cols, feature_j)
    }
  }
  if (length(identical_cols) > 0) {
    identical_cols = list(identical_cols)
    names(identical_cols) = feature_i
    identical_cols_list = c(identical_cols_list, identical_cols)
  }
}

for (i in names(identical_cols_list)) {
  cat("removing ", identical_cols_list[[i]], " because they are the same with ", i, "\n")
  traindf = del_cols(traindf, identical_cols_list[[i]])
  testdf = del_cols(testdf, identical_cols_list[[i]])
}

# save(traindf, file="train04.RData")
# save(testdf, file="test04.RData")
# rm(list=ls())
# gc()
# load("train04.RData")
# load("test04.RData")

# remove linear combos
features = get_features(traindf)
library(caret)

linear_combos = findLinearCombos(traindf[, features])
linear_combos_name = features[linear_combos$remove]
cat("Linear combos\n")
cat(linear_combos_name, "\n")

traindf = del_cols(traindf, linear_combos_name)
testdf = del_cols(testdf, linear_combos_name)

save(traindf, file="train04.RData")
save(testdf, file="test04.RData")
write.csv(traindf, file="train04.csv", row.names=F)
write.csv(testdf, file="test04.csv", row.names=F)
