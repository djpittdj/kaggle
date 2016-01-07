Springleaf contest. Finished at 70 out of 2225 teams (top 3%).
Feature engineering:
1. Occupation features (var0404 & var0493) to group together similar occupations, e.g., "president" and "vice president" are grouped together, all "nurse" are grouped together;
2. Count the number of missing values in each row of the data and use it as a feature;
3. Remove features that have only one unique value and missing values;
4. Use the time difference between var0073 and var0075, as well as between var0204 and var0217 in terms of days;
5. Use only the year and month information in the time features, including "var0073", "var0075", "var0156", "var0157", "var0158", "var0159", "var0166","var0167","var0168","var0169","var0176","var0177","var0178","var0179","var0204","var0217", "var0531";
6. deal with zipcode and city;
7. cut numeric features into categorical variables using quantile;
8. remove one of the two features in which all of their elements are the same;
9. remove features that show linear combinations;
10. calculate the correlation matrix between each feature with the rest features and remove those with high correlations;
11. count the number of times a value appears in a feature and use that as a feature;
