import numpy as np
from sklearn.model_selection import train_test_split

a = np.arange(1, 101)
print(a)

b = np.arange(501, 601)
print(b)

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42, shuffle=False) # if shuffle set to False data will stay ordered (not great)
print(a_train.shape, a_test.shape)
print(a_train, "\n", a_test)
print(b_train.shape, b_test.shape)
print(b_train, "\n", b_test)

