"""
Decision Trees are suitable for classification and regression.
They can work with numerical and categorical features.
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X, y = iris.data, iris.target

clf = DecisionTreeClassifier()

clf = clf.fit(X, y)
clf.predict([[6.1, 2.7, 3.9, 1.2]]) # two-dimensional

plot_tree(clf)

plt.figure(figsize=(15,12))
plot_tree(clf, filled=True,
          feature_names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
          class_names=["Setosa", "Versicolour", "Virginica"])
plt.show()




