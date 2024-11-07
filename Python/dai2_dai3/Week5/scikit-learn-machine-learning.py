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

# the .fit() fit command makes a lot of computations to take place and stores them in attributes
print(model.coef_) # this prints the slope
print(model.intercept_) # this prints the intercept

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
Clusters: The goal of clustering is to maximize the similarity of observations within a cluster and maximize the the dissimilarity between clusters. 
    - Clustering: Unsupervised Learning
            
    - Classification: 

K-Nearest Neighbors (KNN) => Classification method used in supervised learning;
    - It uses data already classified with known labels, and in order to classify new data into a label, it looks at k closest points in the labeled data. So it uses the given labeled data and looks at what's closes to the data we're trying to predict a label for, and assigns the closest label to it.

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
        - The model does not capture the underlying patterns in the data.
        The model has low accuracy on both training and test sets.

    2. Overfitting - (High Variance): A model that’s too complex (like a high-degree curve for simpler data) fits the training data too closely, capturing noise instead of meaningful patterns. This results in high variance (the model overfits the data), where the model performs well on training data but poorly on new data.
        - The model has memorized the training data but performs poorly on new data.
        - The model fits the random noise in the training data.

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
    
    The validation curve summarizes the tradeoff between training and validation errors as we vary the model complexity. The learning curve summarizes the tradeoff between training and validation errors as we vary the size of the training set.
    
    DecisionTreeClassifier vs DecisionTreeRegressor:
        - DTR predicts continuous values (regression), while DTC predicts categorical labels (classification).
        - DTR uses mean squared error (MSE) or mean absolute error (MAE) as the split criterion, whereas DTC uses Gini impurity or information gain.
        - DTR produces a continuous value as the predicted output, while DTC outputs a class label.
        - DTR can handle missing values by imputing them with the mean or median of the respective feature. DTC can also handle missing values, but it will propagate them to the leaf nodes.
        - DTR typically requires tuning parameters like max_depth, min_samples_split, and min_samples_leaf. DTC also has these hyperparameters, plus criterion (Gini or entropy) and class_weight (balanced or not).
    
    When to use each:
        1. Decision Tree Regression (DTR):
            - When the target variable is continuous (e.g., predicting a numerical value).
            - When the problem involves predicting a range or interval (e.g., predicting a continuous value between 0 and 100).
            - When you need to model non-linear relationships between features and the target variable.
        2. Decision Tree Classifier (DTC):
            - When the target variable is categorical (e.g., classifying into two or more classes).
            - When the problem involves predicting a discrete label (e.g., spam/not spam, 0/1).
            - When you need to model complex relationships between features and class labels.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()

### 5.6 - Linear Regression
# Simple example:
rng = np.random.RandomState(1)
x = 10 * rng.rand(50) # array of 50 random numbers from 0 to 10
y = 2 * x - 5 + rng.randn(50) # array of 50 random numbers with an intercept of -5 and a slope of 2
# plt.scatter(x, y)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y) # use a two-dimensional version of x to fit the model

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit , yfit)
plt.show()

# Close values
print("Model slope:     ", model.coef_)
print("Model intercept: ", model.intercept_)


# Making a three-dimensional polynomial basis function:
from sklearn.preprocessing import PolynomialFeatures
x = np.array([2, 3, 4])
poly = PolynomialFeatures(3, include_bias=False)
print(poly.fit_transform(x[:, None]))

# Complex sine model:
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7),
                           LinearRegression())
rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()


"""
### Basis function regression: 
    - It adapts linear regression to handle complex (nonlinear) relationships between the input x and output y.
    => This means that with basis functions, we transform the input with exponents, in order to square or cube, etc the input, so that the regression line doesn't always have to follow a simple straight path, and can adapt to the different curved features of the data.
        e.g. Imagine you throw a ball and you want to follow its path with a line, if you use a simple linear regression line, the line can only move in one direction, thus it won't show in detail changes of the patterns in the data. But by being able to add exponents to the data via basis functions, we can adapt the curvature of the line in order to show a better example of the path that the ball made.

### Regularisation
    - Helps avoid overfitting on basis function regression by penalizing large coefficient values.
        1. Ridge Regression (L2 Regularization):
            ==> Penalizes the sum of the squares of coefficients. Helps keep all coefficients small, making the model more stable and generalizable.
        2. Lasso Regression (L1 Regularization):
            ==> Penalizes the sum of the absolute values of coefficients, often pushing some coefficients to zero. This makes the model simpler by keeping only the most important features. 
"""

# Decision Trees:
# data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit(X, y)
# the python method visualize_classifier is written in notebook 5.8 -> visualize_classifier(DecisionTreeClassifier(), X, y)


# Multiple overfitting estimators combined can reduce the effect of this overfitting (random forests in terms of classification -> categorical variables)
# Bagging makes use of an ensemble of overfitting estimators which overfit the data and averages the results to find the better classification. An ensemble of randomized decision trees is known as a random forest.
# We can to a type of bagging classification (what was just explained) this way:

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
                        random_state=1)

bag.fit(X, y)
# visualize_classifier(bag, X, y) -> method to visualize in 5.8

# ensemble of randomized decision trees:
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
# visualize_classifier(model, X, y); -> def in 5.8


# Random Forest in terms of Regression -> continuous variables
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
# forest.fit(x[:, None], y)

# xfit = np.linspace(0, 10, 1000)
# yfit = forest.predict(xfit[:, None])
# ytrue = model(xfit, sigma=0) -> def defined in 5.8

# plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
# plt.plot(xfit, yfit, '-r');
# plt.plot(xfit, ytrue, '-k', alpha=0.5)
# plt.show()


"""
1. Decision Trees:
    - To make decisions based on features of data. It conditionally creates branches that flow towards different directions depending on the answer.
    It splits the data by asking a binary (yes/no) question on each node (feature-based).
    - Limitations: Overfitting => Decision trees can overfit easily (they memorize data's noise instead of general patterns which makes them less accurate on new data.
    
2. Random Forests: A collection of decision trees
    - A Random Forest is an ENSEMBLE method that combines the predictions of multiple decision trees.
    - It uses multiple trees to reduce overfitting. By making an average of multiple trees, the random forest reduces overfitting risk because the errors of the individual decision trees tend to cancel each other out. Trees are also sensitive to training data, while a small change in a tree can lead to a very different tree, within a random forest which contains multiple trees, a small change doesn't make such an important impact.
    
    How It Works:

Bootstrap Sampling: Each tree is trained on a random subset of the training data (known as bootstrapping). This helps each tree capture different patterns.
Feature Selection for Splits: When each tree is growing, it selects a random subset of features at each split. This way, each tree sees slightly different information and makes slightly different splits.
Prediction: For classification, each tree gives its prediction (e.g., category “high income” or “low income”), and the forest takes a majority vote. For regression, it averages the predictions.
Example:

Suppose you want to classify whether a person will buy a product.
A random forest will build multiple trees. Each tree might make a different decision based on its training subset.
By aggregating the output (e.g., majority vote), you get a more reliable prediction than relying on a single tree.
Technical Aspects in Practice:

Hyperparameters: Important parameters include the number of trees (n_estimators), the max depth of each tree (max_depth), and the number of features considered at each split (max_features).
Scikit-Learn: In Python’s Scikit-Learn, RandomForestClassifier is used for classification tasks, and RandomForestRegressor is used for regression.
Advantages:

Accuracy: Random Forests are often very accurate because they combine many models.
Resistant to Overfitting: By combining multiple trees, random forests reduce the likelihood of overfitting, even with large trees.
Disadvantages:

Interpretability: While a single decision tree is easy to interpret, a random forest with hundreds of trees is more of a “black box.”
Computational Cost: Training multiple trees takes more time and memory, especially for large datasets.

Summary
Random Forests are powerful ensemble models that combine multiple decision trees to make more accurate and robust predictions. By averaging or voting on predictions from many trees trained on different parts of the data, random forests strike a balance between flexibility (reducing overfitting) and complexity, making them highly effective in a wide range of classification and regression tasks.
"""





""" 5.1 K-Means
### K-Means Clustering:
# Another type of unsupervised machine learning models: Clustering algorithms
The k-means algorithm searches for a pre-determined number of clusters within an unlabeled multidimensional dataset.
    - So basically it calculates the distance between each point and the center of the cluster it belongs to, and makes sure that the distance form the center of its cluster to it is smaller than the distance from the point to other cluster centers.
    
"""

# visual example of a k-means algorthim plot:
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

# The K-Means algorithm makes the clusters automatically (scikit-learn using estimator API):
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
# and now visualise results of cluster with its centers:
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()



""" 5.11 K-Means
    - How the K-means algorithm finds these clusters: Expectation-Maximization (E-M):
        - The E-M algorithm consists of:
            1. Guess some cluster centers
            2. Repeat until converged
                2.1. E_Step: Assign points to the nearest cluster center
                2.2. M_Step: set the cluster centers to the mean
                
        Expectation–maximization (E–M) is a powerful algorithm that comes up in a variety of contexts within data science.
*k*-means is a particularly simple and easy-to-understand application of the algorithm, and we will walk through it briefly here.
In short, the expectation–maximization approach here consists of the following procedure:

1. Guess some cluster centers
2. Repeat until converged
   1. *E-Step*: assign points to the nearest cluster center
   2. *M-Step*: set the cluster centers to the mean 

Here the "E-step" or "Expectation step" is so-named because it involves updating our expectation of which cluster each point belongs to.
The "M-step" or "Maximization step" is so-named because it involves maximizing some fitness function that defines the location of the cluster centers—in this case, that maximization is accomplished by taking a simple mean of the data in each cluster.

The literature about this algorithm is vast, but can be summarized as follows: under typical circumstances, each repetition of the E-step and M-step will always result in a better estimate of the cluster characteristics.

We can visualize the algorithm as shown in the following figure.
For the particular initialization shown here, the clusters converge in just three iterations.
"""








































# DATA PREPARATION
import pandas as pd
pd.options.display.max_rows = None
import seaborn as sns
iris = sns.load_dataset('iris')
X = iris[['sepal_width', 'sepal_length', 'petal_width']] # Predictors
y = iris['petal_length'] # Target feature to predict
# MODEL SELECTION AND HYPERPARAMETER SELECTION (MODEL SPECIFIC)
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
print(model)
# List all selected hyperparameters
print(model.get_params(deep=True))
# DERIVE MODEL FROM LABELED DATA (TRAIN MODEL/FIT MODEL)
model.fit(X,y)
# DISPLAY MODEL (MODEL SPECIFIC)
print(model.intercept_, model.coef_)
# VALIDATE MODEL USING LABELED DATA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
# Predict target feature for the labeled data
y_pred = pd.Series(model.predict(X), name='y_pred')
# Calculate the difference between predicted and real values for the labeled data
err = pd.Series(y_pred-y, name='err')
print(pd.concat([y, y_pred, err], axis=1))
# Metrics
mae = mean_absolute_error(y_true=y, y_pred=y_pred)
mape = mean_absolute_percentage_error(y_true=y, y_pred=y_pred)
rmse = root_mean_squared_error(y_true=y, y_pred=y_pred)
print(f'MAE : {mae:.3f} - MAPE : {mape:.3f} – RMSE : {rmse:.3f}')











"""
Hot to choose best model:

To detect overfitting and underfitting, we can experiment with various models and techniques, assessing their performance on both training and test data. Here’s how to approach this:

### 1. **Evaluate with Cross-Validation**
   - **Cross-validation** (e.g., k-fold cross-validation) is a strong tool for assessing a model’s generalization. By splitting the data into multiple subsets, we can train and validate the model multiple times to get an average performance score, reducing the risk of overfitting or underfitting due to a single train-test split.

### 2. **Try Other Models for Comparison**
   In addition to linear regression, we can try different models, including those that may handle complex relationships or reduce overfitting:

   - **Polynomial Regression**: Extends linear regression to handle non-linear relationships. However, it’s prone to overfitting if the polynomial degree is too high.
   - **Ridge and Lasso Regression**: Both are forms of regularized regression that add penalties to reduce overfitting:
     - **Ridge** (L2 regularization) reduces overfitting by adding a penalty for large coefficients.
     - **Lasso** (L1 regularization) reduces overfitting and can also perform feature selection by driving some coefficients to zero.
   - **Elastic Net**: Combines L1 and L2 penalties for more balanced regularization, often useful if Lasso or Ridge alone isn’t performing optimally.

   You could also consider tree-based models, which often handle non-linear relationships better than simple linear models:
   - **Decision Trees**: Basic tree models can help identify non-linear patterns but are prone to overfitting, especially if not pruned.
   - **Random Forest**: An ensemble method that averages multiple decision trees to reduce overfitting.
   - **Gradient Boosting**: Another ensemble technique that sequentially builds trees, which can be powerful but also prone to overfitting if not tuned carefully.

### 3. **Compare Train and Test Performance**
   Check if your model performs well on training data but poorly on test data. Key indicators:
   - **Underfitting**: If both training and test errors are high, the model might be too simple for the data.
   - **Overfitting**: If training error is low but test error is high, the model might be too complex and is overfitting the training data.

### 4. **Use Validation Curves**
   Validation curves are plots that show the relationship between a model parameter (like polynomial degree, or regularization strength in Ridge/Lasso) and the model's performance on the train and test sets. This helps find the optimal complexity.

### 5. **Regularization for Overfitting Control**
   Regularization is a key technique to control overfitting. Here’s how to use it with Ridge and Lasso:

   ```python
   from sklearn.linear_model import Ridge, Lasso
   from sklearn.model_selection import cross_val_score

   # Ridge Regression
   ridge_model = Ridge(alpha=1.0)  # adjust alpha to control regularization strength
   ridge_cv_score = cross_val_score(ridge_model, X, y, cv=5)
   print("Ridge CV Score:", ridge_cv_score.mean())

   # Lasso Regression
   lasso_model = Lasso(alpha=0.1)  # adjust alpha similarly
   lasso_cv_score = cross_val_score(lasso_model, X, y, cv=5)
   print("Lasso CV Score:", lasso_cv_score.mean())
   ```

### 6. **Learning Curves**
   A learning curve shows model performance on training and validation sets as the training size increases. If the training score is high and the test score is low, the model is overfitting. If both are low, the model is underfitting.

### Summary Steps

1. Start with cross-validation on multiple models.
2. Use regularized regression (Ridge, Lasso, Elastic Net).
3. Compare train-test errors.
4. Look at learning and validation curves to assess fitting issues.
5. Choose the model with the best balance of train-test performance.
"""