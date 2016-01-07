for (i in c(404, 493)) {
  feature = paste0("var0", i)
  data_var = c(as.character(traindf[,feature]), as.character(testdf[,feature]))
  data_var[data_var == ""] = -1
  data_var = tolower(data_var)
  write.table(data_var, file=paste0(feature, ".dat"), row.names=F, col.names=F, quote=F)
  sorted = sort(table(data_var[data_var!="-1"]), decreasing=T)
  sorted = data.frame(sorted)
  sorted["occupation"] = rownames(sorted)
  rownames(sorted) = 1:nrow(sorted)
  names(sorted)[1] = "count"
  write.csv(sorted, file=paste0(feature, "_table.csv"), row.names=F)
}


