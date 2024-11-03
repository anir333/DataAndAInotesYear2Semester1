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

# to know all seaborn available datasets:
sampledatasets = sns.get_dataset_names()
print(sampledatasets)

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
from sklearn.model_selection import train_test_split, LeaveOneGroupOut

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
# here X1 is half of the data used for training and X2 is the other half used for testing
# y1 contains target labels for X1, y2 contains target labels for X2
print("\n\nX1\n: ", X1)
print("\n\ny1\n: ", y1)
print("\n\nX2\n: ", X2)
print("\n\ny2\n: ", y2)


# fit the model on one set of data
model.fit(X1, y1)

from sklearn.metrics import accuracy_score
# evaluate the model on the second set of data
y2_model = model.predict(X2)
print("\n\nCorrect way accuracy score: ", accuracy_score(y2, y2_model)) # It shows a more reasonable representation of an accuracy score of the model



# Problem: If we divide the data in training dataset and testing dataset, we are losing some data that we could've used to train our model.
# Solution: Cross-Validation: a sequence of fits where each subset of the data is used both as a training and as a validation set.
y2_model = model.fit(X1, y1).predict(X2) # we use first X1 as data, and y1 as training data to redict X2
y1_model = model.fit(X2, y2).predict(X1) # now we use X2 as training data and y2 as test data to predict X1

print("Accuracy model y2 prediction: ", accuracy_score(y2, y2_model)) # pass in the targets and the model trained
print("Accuracy model y1 prediction: ", accuracy_score(y1, y1_model))

#  Since each model is trained/tested on half of the data, we can combine them by taking the mean of their accuracy in order to get a better measure of their performance. This way of cross-validation is known as two-fold cross-validation since we spilt the data in two sets and use each in turn as a validation set.

# In order to make even more splits on our data and make use each of them to evaluate the model fit, we can do it faster and more easily this way:
from sklearn.model_selection import cross_val_score
print(cross_val_score(model, X, y, cv=5)) # we're telling it to make 5 splits, train them, test them, and give us the accuracy | (X is the data and y is the target)
# So it will return an array of 5 different accuracies


""" LOO (Leave-one-out cross-validation):
        - Create a fold for each data point in the dataset.
        - For each fold, train the model on ALL data points except one, and then test the model on that single data point left out. Repeat this process for every single data point.
"""
from sklearn.model_selection import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut())
print(scores) # an array of 1's or 0's for every single data point in the data provided (1 is True: model predicted correctly, 0 is False: model predicted incorrectly)

# Since in this case we have 150 data points, we wil have 150 samples of accuracy, by getting the mean we can get the total model accuracy:
print(scores.mean())




"""
The Bias-Variance Trade-off explains the balance needed to build a good model that generalizes well:

    1. Underfitting - (High Bias): A model that’s too simple (like a straight line for complex data) won’t capture the patterns in the data, leading to high bias (the model underfits the data). It ignores important data features, resulting in a poor fit.

    2. Overfitting - (High Variance): A model that’s too complex (like a high-degree curve for simpler data) fits the training data too closely, capturing noise instead of meaningful patterns. This results in high variance (the model overfits the data), where the model performs well on training data but poorly on new data.

        - The goal is to find a balance—enough flexibility to capture patterns without fitting noise—achieving a model with low bias and low variance.
    
    - For high-bias models, the performance of the model on the testing/validation set is similar to the performance on the training set.
    - For high-variance models, the performance of the model on the testing/validation set is far worse than the performance on the training set.
    
    - The training score is everywhere higher than the validation score. This is generally the case: the model will be a better fit to data it has seen than to data it has not seen.
    
    - For very low model complexity (a high-bias model), the training data is underfit:
            - The model is a poor predictor both for the training data and for any previously unseen data.
    - For very high model complexity (a high-variance model), the training data is overfit: 
            - The model predicts the training data very well, but fails for any previously unseen data.
            
    - For some intermediate value, the validation curve has a maximum. This level of complexity indicates a suitable trade-off between bias and variance.
    
    # The training score measures the model's accuracy on the training data.
    # The validation score measures the model's accuracy on unseen data.
    
        - Examining the relationship between the training score and the validation score can give us useful insight into the performance of the model.
    
    # With more data, a more complex model can fit well without overfitting, as shown by closer training and validation scores. A learning curve shows how training and validation scores change as the dataset grows:
        - Small datasets lead to overfitting (high training, low validation scores).
        - Large datasets lead to underfitting (training score decreases, validation score increases).
        
    # When the learning curve has already converged (when training and validation curves are already close to each other), adding more data won't significantly improve the fit.
    # To improve the final score, we can use a more complex model, though it may increase variance between training and validation scores.
"""

