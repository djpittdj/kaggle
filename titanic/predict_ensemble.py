#!/usr/bin/python
from __future__ import division, print_function
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

# from Trevor Stephens" tutorial and its comment
def combine_titles(title):
	if title == "Mlle":
		return "Miss"
	elif title == "Mme":
		return "Mrs"
	elif title in ["Capt", "Don", "Major", "Sir", "Jonkheer"]:
		return "Sir"
	elif title in ["Dona", "Lady", "the Countess"]:
		return "Lady"
	else:
		return title

def clean(df):
	titles = df.Name.apply(lambda x: x.split(",")[1].split(".")[0].lstrip())
	titles = titles.apply(combine_titles)
	df["Title"] = titles

	df.Fare = df.Fare.map(lambda x: np.nan if x==0 else x)
	fare_median_table = df.pivot_table("Fare", index="Pclass", aggfunc="median")
	df.Fare = df[["Fare", "Pclass"]].apply(lambda x: fare_median_table[x["Pclass"]] if pd.isnull(x["Fare"]) else x["Fare"], axis=1 )

	age_median_table = df.pivot_table("Age", index="Sex", aggfunc="median")
	df.Age = df[["Age", "Sex"]].apply(lambda x: age_median_table[x["Sex"]] if pd.isnull(x["Age"]) else x["Age"], axis=1)

	df.Cabin = df.Cabin.fillna("X")
	df["Deck"] = df.Cabin.apply(lambda x:x[0])

	df["Family_Size"]=df["SibSp"]+df["Parch"]

	most_embarked = mode(df.Embarked)[0][0]
	df.Embarked = df.Embarked.fillna(most_embarked)

	df["Age"] = pd.qcut(df["Age"], 7, labels=range(7))
	df["Fare"] = pd.qcut(df["Fare"], 10, labels=range(10))

	df["Sex"] = df["Sex"].map({"female":0, "male":1})
	df["Embarked"] = df["Embarked"].map({"S":0, "C":1, "Q": 3})
	
	titles_dict = {}
	for i, title in enumerate(["Mr", "Miss", "Mrs", "Master", "Dr", "Rev", "Sir", "Col", "Lady", "Ms"]):
		titles_dict[title] = i
	df["Title"] = df["Title"].map(titles_dict)

	deck_dict = {}
	for i, deck in enumerate(["X", "C", "B", "D", "E", "A", "F", "G", "T"]):
		deck_dict[deck] = i
	df["Deck"] = df["Deck"].map(deck_dict)

	return df

traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")
num_train = len(traindf)
testdf["Survived"] = np.nan
combinedf = pd.concat([traindf, testdf], ignore_index=True)
cleandf = clean(combinedf)
cleandf.drop(["Cabin", "Ticket", "Name"], axis=1, inplace=True)
clean_traindf = cleandf.ix[:num_train-1]
clean_testdf = cleandf.ix[num_train:]

predictors = ["Age", "Embarked", "Fare", "Pclass", "Sex", "Title", "Deck", "Family_Size"]
for func_name in [RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier]:
	clf = func_name(n_estimators=2000)
	clf.fit(clean_traindf[predictors], clean_traindf["Survived"])
	predicted = DataFrame({"Survived" : clf.predict(clean_testdf[predictors]).astype(int), "PassengerId" : testdf["PassengerId"]})
	predicted.to_csv("predicted_%s.csv" % func_name.__name__, index=False)
