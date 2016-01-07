library(sqldf)

del_cols = function(df, cols_to_remove) {
  return (df[,!(names(df) %in% cols_to_remove)])
}

get_features = function(df) {
  features = names(df)
  features = features[!(features %in% c("id", "target"))]
  return (features)
}

# 1-way count
count1way = function(input_df, v) {
  df = data.frame(f1=input_df[,v])
  sum1 = sqldf("select f1, sum(1) as cnt from df group by 1")
  tmp = sqldf("select cnt from df a left join sum1 b on a.f1 = b.f1")
  tmp$cnt[is.na(tmp$cnt)] = 0
  return(tmp$cnt)
}
