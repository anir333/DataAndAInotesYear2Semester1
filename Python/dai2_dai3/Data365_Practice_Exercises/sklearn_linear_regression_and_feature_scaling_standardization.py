import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Sklearn_Linear_Regression/Simple linear regression Dataset/1.01. Simple linear regression.csv')
print(data.head())

X = data['SAT']  # independent variable (input feature)
y = data['GPA']  # dependent variable (output target)

# They are both vectors of length 84 (one dimensional arrays)
print(X.shape)
print(y.shape)
X_matrix = X.values.reshape(-1, 1)
# X_matrix = pd.DataFrame(X, columns=['SAT'])

# Regression:
reg = LinearRegression()
"""
    - Hypermarameters:
        normalize=True
        copy_X=True -> copies the input before fitting them
        fit_intercept=True
        n_jobs=1 -> only one cpy is used (if lots of data you can do 2 or more)
"""
reg.fit(X_matrix, y)  # input and target

"""
- Standardization: The process of subtracting the mean and dividing by the standard deviation ==> A type of normalization
- Normalization: Subtract the mean but divide by the L2-norm of the inputs
"""

print(f'\nR2 -> R-squared: {reg.score(X_matrix, y)}') # this is the R square value not the adjusted one
# The adjusted r-squared is better for a regression model
print(f'Coefficients (slope): {reg.coef_}')
print(f'Intercept: {reg.intercept_:.3}')

# Making predictions:
# Predict takes as arguments the input we want to predict and outputs the prediction by the model
print(reg.predict(np.array([1740])[:, np.newaxis]))

new_data = pd.DataFrame(data=[1740, 1760]) # SAT
print(new_data)
print(reg.predict(new_data)) # predicts GPA

new_data['Predicted_GPA'] = reg.predict(new_data)
print(new_data)

# Plot the regression:
plt.scatter(X, y)

"""
This equation is a way to use a trained linear regression model to make predictions on new data. Here’s how each part of the equation plays a role:

1. **`reg.coef_ * X_matrix`**: 
   - This term represents the main part of the prediction. The `coef_` (coefficients) are learned weights that indicate the relationship between each feature in `X_matrix` and the target variable (`y_value`).
   - By multiplying each feature value in `X_matrix` by its corresponding coefficient in `reg.coef_`, we calculate the contribution of each feature to the prediction.

2. **`+ reg.intercept_`**:
   - The `intercept_` is a constant value added to the prediction. It shifts the line up or down and accounts for the baseline level of the target variable when all feature values are zero.

In brief, `y_value = reg.coef_ * X_matrix + reg.intercept_` combines the learned coefficients and intercept to compute the output of the linear regression model for a given input, `X_matrix`. This is how we apply the model to make predictions based on the relationship it learned during training.
"""
y_value = reg.coef_ * X_matrix + reg.intercept_
plt.plot(X, y_value, lw=4, c='orange', label='Regression Line')
sns.set_style("darkgrid")
plt.xlabel('SAT', fontsize=20)
plt.ylabel("GPA", fontsize=20)
plt.show()



"""
Standardization - Feature Scaling

    What is the purpose of standardizing data in machine learning?
        - To ensure each feature contributes equally to the model by bringing them to the same scale.
"""
scaler = StandardScaler()
# scaler.fit(pd.DataFrame(X))
x_scaled = scaler.fit_transform(pd.DataFrame(X)) # fits the data and subtracts the mean and divides by the std to standardize the data
print(x_scaled)


# Regression with scaled features:
reg = LinearRegression()
reg.fit(x_scaled, y)
print(reg.coef_)
print(reg.intercept_)

# Summary table: The smaller the weight the smaller its impact, the bigger the weight the bigger the impact, in this case I don't have another columns to see the weight of, but you compare them
reg_summary = pd.DataFrame([['Intercept/Bias'], ['SAT']], columns=['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0]
print(reg_summary)


"""

"""