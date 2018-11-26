# Logistic Regression

# Importing the libraries
import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd

# Importing the dataset
import scipy.io
mat = scipy.io.loadmat('Sample 1.mat')
X = mat['input']
y = mat['target']

count = 0
i = 0

while(True):
    i += 1
    if count == 1100:
        break;
    if y[i][0] == 0:
        count += 1
        X = np.delete(X, (i), axis=0)
        y = np.delete(y, (i), axis=0)
        i -= 1

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, :])
X[:, :] = imputer.transform(X[:, :])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

##classifier = linear_model.Lasso(alpha = 0.1)

# Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#y_pred = np.reshape(y_pred, (446, 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)