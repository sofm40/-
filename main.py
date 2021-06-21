import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVR
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
nfold = 3
kf = KFold(n_splits=nfold, shuffle=True, random_state=2020)
prediction1 = np.zeros((len(X_test_c), ))
i = 0
for train_index, valid_index in kf.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, label_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, label_valid = X_train_c[valid_index],y_train_c[valid_index]
    clf=SVR(kernel='rbf',C=1,gamma='scale')
    clf.fit(X_train,label_train)
    x1 = clf.predict(X_valid)
    y1 = clf.predict(X_test_c)
    prediction1 += ((y1)) / nfold
    i += 1
result1 = np.round(prediction1)
id_ = range(210,314)
df = pd.DataFrame({'ID':id_,'CLASS':result1})
df.to_csv("baseline.csv", index=False)
