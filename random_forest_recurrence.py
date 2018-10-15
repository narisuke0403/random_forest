"""
回帰分析（Scikit-learn)
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

boston = datasets.load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
reg = RandomForestRegressor().fit(X_train, y_train)
print('  score_train: {}'.format(reg.score(X_train, y_train)))
print('  score_test: {}'.format(reg.score(X_test, y_test)))
