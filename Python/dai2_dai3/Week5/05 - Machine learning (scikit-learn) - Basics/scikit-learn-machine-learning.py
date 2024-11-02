"""
Categories of machine learning:
    1. Supervised learning
        - Modeling the relationships between measured features of data and some labels associated with it. So in machine learning, the algorithm implemented learns about the different measurable data given, and its labeled association, so that it can later on predict future labels with some measurable data given.
            e.g. Finding specific correlations between height/weight and the age group as the label, so that the algorithm can `predict mathematically` the age group given some height and weight data provided.
        - It can be further subdivided:
            1.1. Classification: Labels are discrete categories
                - In classification the labels are distinct to each other, they do not overlap, and they are usually specific categories. Classification tasks involve predicting labels from a fixed set of categories.
                    e.g. An email is classified either spam or not spam, it can't be classified as both at the same time.

            1.2. Regression: Labels are continuous quantities/values
                 - In regression, the output can take a wide range of values, often numerical and not limited to a set of categories. The model's output is typically a single continuous value.
                    e.g. Prediction of house pricing given the square footage, number of rooms, location, etc. THis value can have a wide range depending on the factors given. Another example would be temperature forecasting, which analyzes past data to predict future temperature (within a realistically possible human range).

    2. Unsupervised learning
        - Modeling the features (values/data) of a dataset without reference to any label, in order to find hidden patterns/groupings or structures in data.
        - Includes:
            2.1. Clustering algorithms that identify distinct groups of data.
            2.2. Dimensionality reduction algorithms which search to reduce number of features/data in order to retain the essential information.
"""

# We have a dataframe (2D two-dimension array) with features (columns) and samples (rows)
# n_features is the number of columns (labels)
# n_samples is the number of rows (data)
# What we do is that we take a target array saved by convenience in a variable called y
# Then we remove the target array from the rest of that data, and we save the rest of the features/samples in an array by convenience called x

# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species', height=1.5)
plt.show()

# Extract feature matrix (matrix with independent/predictor variables)
# (by deleting the target variable)
X_iris = iris.drop('species', axis=1)
print(X_iris.shape)
print(X_iris)

y_iris = iris['species']
print(y_iris.shape)
print(y_iris)

# Process of steps to follow when using Scikit_Learn Estimator API

# 1. Choose a class of model (a class of model is not the same as an instance of a model)
from sklearn.linear_model import LinearRegression

# 2. Choose model hyperparameters
model = LinearRegression(fit_intercept=True) # Call the LinearRegression class and set in the hyperparameter fit_intercept to True to make it more fitting to real life data

# 3. Arrange data into a features matrix and target vector:
# data:
rng = np.random.RandomState(42)
x = 10 * rng.rand(50) # 50 random numbers between 0 and 10
y = 2 * x - 1 + rng.randn(50) # slope of 2 and intercept -1 + random numbers to make it more realistic

# y was created as a 1-dimensional Numpy array and hence ready to use with scikit-Learn
print(type(y))
print(y.shape)
print(y)

# x was also created as an 1-dimensional Numpy array
print(type(x))
print(x.shape)
print(x)

# But we need a two-dimensional data structure as feature matrix
# The following codes transforms an array into a single column matrix.
X = x[:, np.newaxis]
print(X)
print(X.shape)
print(type(X)) # X.__class__

# mind the difference:
# x (lowercase) is a 1-dimensional Numpy array (vector) with 50 elements
# (displays as a list)
print(x.shape)
# X (uppercase) is a 2-dimensional Numpy array (matrix) with 50 rows and 1 column
# (displays as q list of lists)
print(X.shape)

# 4. Fit model to the data (apply the model to our data):
model.fit(X, y) # we use the two-dimensional version of x, (X), otherwise we'll get error at runtime

# the fit command makes a lot of computations to take place and stores them in attributes
print(model.coef_)
print(model.intercept_)

# Predict labels for unknown data
# new data that wasn't used for training the model
xfit = np.linspace(-1, 11) # linspace creates by default 50 numbers, evenly distributed, and in this case between tha range of -1 and 11
print(xfit)

# n_samples, n_features features matrix (two-dimensional)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit) # make prediction

# let's visualise the data and the model fit:
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()

# Supervised Learning example:
# Let's divide the data in training and test data:
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1) # give it the dataframe data to train and the target data, so it knows that we are training it to predict the target array daya given the training data provided, by default it does 75% data for training and 25% for testing
print(X_iris, y_iris)

# With the data arranged we follow our steps to predict the labels:
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data

# We can use the accuracy_score to see the fraction of correctly predicted labels
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model)) # over 97%

# Unsupervised Learning Example
from sklearn.mixture import GaussianMixture      # 1. Choose the model class
model = GaussianMixture(n_components=3,
                        covariance_type='full')  # 2. Instantiate the model
model.fit(X_iris)                                # 3. Fit to data
y_gmm = model.predict(X_iris)                    # 4. Determine labels

iris['cluster'] = y_gmm
# and now to see the plot check notebook 5.2, after the previous line

# ----------------------------------------------------------------------------------------------------------
# Hyperparameters and Model Validation

"""
K-Nearest Neighbors (KNN) => Classification method used in supervised learning;
    - It uses data already classified with known labels, and in order to classify new data into a label, it looks at k closest points in the labeled data. So it uses the given labeled data and looks at what's closes to the data we're trying to predict a label for, and assigns the closes label io it.

K-Means is used for unsupervised learning:
    - The data has no labels, thus it groups points into clusters based on similarity of the data without predicting any labels.

Summary:
    - KNN is about classifying based on nearest labeled points, while K-Means is about grouping unlabeled points into clusters.




## Wrong way of doing model validation:
    - Training data and then testing the prediction without dividing the data with a training dataset and a test dataset. If we don't divide the dataset into training/testing, we will get 100% accuracy on the accuracy score when testing the data because the model keeps remembers the training data and compares it with the testing data, and if it finds it present it will simply always give the correct answer.

## Correct way of model validation: Holdout Sets
    - Separate the data and use it to train/test the data in an accurate way:
"""

from sklearn.datasets import load_iris
# data:
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)  # using the KNeighbors classifier model

from sklearn.model_selection import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0,
                                  train_size=0.5)

# fit the model on one set of data
model.fit(X1, y1)

from sklearn.metrics import accuracy_score
# evaluate the model on the second set of data
y2_model = model.predict(X2)
print("\n\nCorrect way accuracy score: ", accuracy_score(y2, y2_model)) # It shows a more reasonable representation of an accuracy score of the model



















