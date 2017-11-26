# do categorical features smoothing
# by changing the categorival feature into
# a probability ranged from 0 to 1
# And use PCA to eliminate features

# stratifiedkfold

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.columns

test_id = test.id
test = test.drop(['id'], axis=1)
train_target = train.target
train = train.drop(['id', 'target'], axis=1)

print 'train.shape: ' + str(train.shape) + ' | test.shape: ' + str(test.shape)

# drop all the 'calc' features

non_calc_col = [col for col in train.columns if not col.startswith('ps_calc')]

train = train[non_calc_col]
test = test[non_calc_col]  # 37 variables

# smoothing transformation

cat_col = [col for col in train.columns if 'cat' in col]
temp_train = train
temp_train['target'] = train_target.values
prior_prob = temp_train.target.mean()
# loop every column
smoothing = 1

print 'Before transformation, train.shape is: ' + str(train.shape)
print 'Before transformation, test.shape is: ' + str(test.shape)

for col in cat_col:
    cat_name = str(col)+'_smooth'
    prob = temp_train.groupby(col, as_index=False)['target'].mean()
    prob['counts'] = temp_train.groupby(col, as_index=False)['target'].count()['target']
    # compute lambda
    prob['smoothing'] = 1 / (1 + np.exp(-(prob['counts'] - 1) / smoothing))
    prob[cat_name] = prob['smoothing']*prob['target'] + \
        (1-prob['smoothing'])*prior_prob
    probs_frame = prob[[col, cat_name]]
    # merge it with the training and test set
    train = pd.merge(train, probs_frame, how='left', on=col)
    train.drop([col], axis=1, inplace=True)
    test = pd.merge(test, probs_frame, how='left', on=col)
    test.drop([col], axis=1, inplace=True)
# drop target

train = train.drop(['target'], axis=1)

print 'After transformation, train.shape is: ' + str(train.shape)
print 'After transformation, test.shape is: ' + str(test.shape)

# transformation complete

# use pca to get first top 20 features

pca = PCA(n_components=20).fit(train)

train_pca = pca.transform(train)
test_pca = pca.transform(test)

# use stratifiedkfold(n_splits=3) to fit_predict the test set 3 times
# and get the proba

lgb_params = dict()
lgb_params['learning_rate'] = 0.01
lgb_params['n_estimators'] = 1000
# lgb_params['max_depth'] = 10
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500

lgb = LGBMClassifier(**lgb_params)

skf = StratifiedKFold(n_splits=3, shuffle=True)

predictions = np.zeros((test_pca.shape[0], 3))
for train_index, test_index in skf.split(train_pca, train_target):
    i = 0
    lgb_train = train_pca[train_index]
    lgb_target = train_target[train_index]
    lgb.fit(lgb_train, lgb_target)
    y_pred = lgb.predict_proba(test_pca)[:, 1]
    predictions[:, i] = y_pred
    i += 1

# write the result to a csv

res = pd.DataFrame()
res['id'] = test_id
res['target'] = predictions.mean(axis=1)
res.to_csv('smooth_pred.csv', index=False)


