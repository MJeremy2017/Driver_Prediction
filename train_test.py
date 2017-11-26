# Steps
#
# 1. Impute missing values, delete some missing rows
# 2. Make a balanced data

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
import sklearn.metrics as m
import pandas as pd
import numpy as np


train3 = pd.read_csv('~/workFiles/RData/SafeDriverPrediction/train3.csv')

train3.shape

# normalized gini index


def gini(list_of_values):
    sorted_list = sorted(list(list_of_values))
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2
    return (fair_area - area) / fair_area


def normalized_gini(y_true, y_pred):
    normalized = gini(y_pred) / gini(y_true)
    return normalized


mc = m.make_scorer(normalized_gini)

X_train, X_test, y_train, y_test = train_test_split(
    train3.drop(['id', 'target'], axis=1), train3.target, test_size=0.3)

# try random forest first

clf = RandomForestClassifier(n_estimators=300)

clf_fit = clf.fit(X_train, y_train)

y_proba = clf_fit.predict_proba(X_test)

y_pred = clf_fit.predict(X_test)

# normalized_gini(y_proba, y_test)  # problems

m.matthews_corrcoef(np.array(y_test), y_pred)

m.accuracy_score(np.array(y_test), y_pred)

m.recall_score(np.array(y_test), y_pred)

m.precision_score(np.array(y_test), y_pred)

m.roc_auc_score(np.array(y_test), y_pred)

# bad result





