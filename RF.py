from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
import keras.backend as K
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomThreshold(BaseEstimator, ClassifierMixin):
    """ Custom threshold wrapper for binary classification"""
    def __init__(self, base, threshold=0.5):
        self.base = base
        self.threshold = threshold
    def fit(self, *args, **kwargs):
        self.base.fit(*args, **kwargs)
        return self
    def predict(self, X):
        return (self.base.predict_proba(X)[:, 1] > self.threshold).astype(int)

dataset_clinical = np.loadtxt("/home/nikhil/Desktop/Project/nik/Code/Submodels/CNN/Hidden Layer Data/stacked_metadata1.csv",delimiter=",")
X = dataset_clinical[:,0:450]
Y = dataset_clinical[:,450]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)
rf = RandomForestClassifier(n_estimators=10).fit(X,Y)  
clf = [CustomThreshold(rf, threshold) for threshold in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]

for model in clf:
    print(confusion_matrix(y_test, model.predict(X_test)))
for model in clf:
    print(confusion_matrix(Y, model.predict(X)))

#print(confusion_matrix(Y, clf.predict(X)))
#assert((clf[1].predict(X_test) == clf[1].base.predict(X_test)).all())
#assert(sum(clf[0].predict(X_test)) > sum(clf[0].base.predict(X_test)))
#assert(sum(clf[2].predict(X_test)) < sum(clf[2].base.predict(X_test)))





