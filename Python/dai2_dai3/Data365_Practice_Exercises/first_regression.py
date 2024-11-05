"""
Linear Regression:
    Prediction of a dependent variable (y) based on an independent variable (x). The regression model is a linear approximation of the function that predicts a dependent variable based on the independent variable.

Correlation:
    Measures the degree of relationship between two variables. -> Correlation doesn't imply causation
    - Shows how they change based on each other.
    - Correlation shows a single point on the plot.
    - Correlation is symmetrical regarding both variables.
     - The x and y variables tend to move in the same direction.

Regression:
    Depicts how one variable affects the change of another variable.
    - It shows the cause and effect.
    - Regression shows as a line that goes through the points and minimizes the distance between them.
    - Regression goes one way, it is not symmetrical regarding the change of both variables.


The p-value suggests how significantly different the intercept coefficient is from 0, a p-value higher than 0.05 means it's very different from 0 (no rejecting null hypothesis) & viceversa
What does a p-value of 0.000 suggest about the coefficient (x)? It is significantly different from 0.




"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/home/anir333/Desktop/KDG/Subjects/Year_2/Semester_1/DataAndAI/Python/dai2_dai3/Data365_Practice_Exercises/First_Regression/1.01. Simple linear regression.csv')
print(type(data))
print(data.head())
print(data.describe())

y = data[['GPA']]
x1 = data[['SAT']]

plt.scatter(x1, y)
plt.xlabel("SAT", fontsize=20)
plt.ylabel("GPA", fontsize=20)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x1, y, train_size=0.8, random_state=42)
linear_regression_model = LinearRegression().fit(X_train, y_train)
model_predict = linear_regression_model.predict(X_test)
plt.scatter(x1, y)
plt.plot(X_test, model_predict, linewidth=2)
plt.show()
print(f'R-squared          : {r2_score(y_test, model_predict):.3f}')
print(f'Mean Squared Error : {mean_squared_error(y_test, model_predict):.3f}')
print(f'Intercept          : {linear_regression_model.intercept_[0]:.3f}')
print(f'Coefficients       : {linear_regression_model.coef_}')

X_test = pd.DataFrame(np.linspace(x1.min(), x1.max(), 500), columns=['SAT'])
print(X_test.head())
y_test = LinearRegression().fit(x1, y)
y_test = y_test.predict(X_test)
plt.scatter(x1, y)
plt.plot(X_test.to_numpy(), y_test, linewidth=2)
plt.show()


"""
WAYS TO CHANGE DATA:
"""
# print(data)
# data.loc[data['GPA'] < 3] = 'No'
# data.loc[data.loc[:, 'GPA'] > 3] = 'Yes'
# print(data)

# Correctly
# data['GPA'] = data['GPA'].apply(lambda x: 'No' if x < 3 else ('Yes' if x > 3 else x))
# print(data)

# Mapping data
# data['GPA'] = data['GPA'].map({'Yes':1, 'No':0})
# print(data)




