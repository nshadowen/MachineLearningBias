# coding: utf-8

# 
# http://blog.fastforwardlabs.com/2017/03/09/fairml-auditing-black-box-predictive-models.html
#     
#     




print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

dataset = pd.read_csv('propublica_data_for_fairml.csv')
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11]].values
y = dataset.iloc[:, 0].values

datasets = [(X,y)]


# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)

    
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(name,score)


get_ipython().run_line_magic('md', '')

No scaling:
    
Automatically created module for IPython interactive environment
Nearest Neighbors 0.632644795464
Linear SVM 0.658161198866
RBF SVM 0.665856622114
Decision Tree 0.67922235723
Random Forest 0.676792223572
Neural Net 0.682462535439
AdaBoost 0.681652490887
Naive Bayes 0.659376265695
QDA 0.659781287971    




from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import numpy as np

grid = {'C':np.logspace(-3,7,15),'gamma':np.logspace(-5, 5, 10)}
clf = GridSearchCV(SVC(), param_grid=[grid],cv=10,n_jobs=4)
clf.fit(X, y)

print ("crossval score = ",clf.best_score_)



