import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
# from pylightgbm.models import GBMClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from rgf.sklearn import RGFClassifier  # warning


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

id_test = test['id'].values

col = [c for c in train.columns if c not in ['target', 'id']]

col = [c for c in col if not c.startswith('ps_calc_')]
# all columns not startwith ps_calc_

print col

train = train.replace(-1, np.NaN)  # this will help exclude nan values
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if
           c not in ['id', 'target']}  # get all the keys of columns


# transformation:
# 1. create a cross feature
# 2. create a feature count all the missing values
# 3. for continuous and ordinal variables, create features over median and mean
# 4. for categorical variables, do one-hot-encoder

def transform(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id', 'target']]
    # create a new feature 'ps_car_13' * 'ps_reg_03'
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    # count the missing values for all rows
    df['negative_one_vals'] = np.sum((df[dcol] == -1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:  # if not binary values
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        # for categorical variables but not binary do one-hot transformation
        if 2 < len(one_hot[c]) < 7:  # not too many categories
            for val in one_hot[c]:
                # np.array has attribute astype
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


train = transform(train)
test = transform(test)

train.shape  # (595212, 186)
print train.columns

col = [c for c in train.columns if c not in ['id', 'target']]
col = [c for c in col if not c.startswith('ps_calc_')]

# Return boolean Series denoting duplicate rows,
# optionally only considering certain columns, keep = False, all duplicated rows
# are labeled true

dups = train[train.duplicated(subset=col, keep=False)]

print dups.columns  # has all columns as col

# eliminate duplicates

train = train[~(train['id'].isin(dups['id'].values))]

train.shape  # (595200, 186)

target_train = train['target']
train = train[col]  # 130 features, features start with 'ps_calc_' are excluded
test = test[col]
print(train.values.shape, test.values.shape)

# ((595200, 130), (892816, 130))

# build a class to run stratified training

# this is a very weired method!
# final training set with features of every model of
# predictions and label is actual value
# final test set are also predictions
# It's using predictions on train set to predict predictions on test set


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        # stratified returns indices for train and test set
        # list makes them tuples, tuple1(trainIndex, testIndex)
        # tuple2(trainIndex, testIndex)
        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                     random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        # enumerate loop index and elements in list
        for i, clf in enumerate(self.base_models):
            # for every classifier
            S_test_i = np.zeros((T.shape[0], self.n_splits))
            # for every fold store the result on true test set into S_test_i

            for j, (train_idx, test_idx) in enumerate(folds):  # n_splits folds
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]  # hold-out test set

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                y_pred = clf.predict_proba(X_holdout)[:, 1]

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:, 1]
            # for each model store mean prediction
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:, 1]
        return res


# run model with tuned parameters

lgb_params = dict()
lgb_params['learning_rate'] = 0.01
lgb_params['n_estimators'] = 300
# lgb_params['max_depth'] = 10
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500

xgb_params = dict()
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.02
xgb_params['n_estimators'] = 300
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9


lgb_model = LGBMClassifier(**lgb_params)

xgb_model = XGBClassifier(**xgb_params)

gb_model = GradientBoostingClassifier(max_depth=5)

log_model = LogisticRegression()

# make prediction

stack = Ensemble(n_splits=3,
                 stacker=log_model,
                 base_models=(lgb_model, xgb_model, gb_model))

y_pred = stack.fit_predict(train, target_train, test)

y_pred.shape

sub = pd.DataFrame()

sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('driver_pred.csv', index=False)









