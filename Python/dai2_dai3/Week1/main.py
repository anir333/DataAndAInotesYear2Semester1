# print("hello world")
#
# " load data "
#
# "make scatterplot"
# "train neural network"
# "confusion matrix"
# "estimate new data"


import numpy as np

# a = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
# b = np.arange(1, 10).reshape(3, 3)
#
# print(a[1, :2])
# print(b)
#
# c = np.array([range(i, i+10) for i in range(10)])
# print(c)
#
# d = np.zeros(2, dtype=float)
# print(d)
#
# from sys import getsizeof
#
# b =  1234567890123456789
# c = 1
# print(getsizeof(b), " ", b, " ", getsizeof(c), " ", c)

# a = np.ones(4).reshape(2, 2)
# print(a)
# b = np.arange(8).reshape(4, 2)
# print(b)
# print(a+b)

arr = np.array([range(1, 31)]).reshape(6, 5)
print("-----------------------")
print(' Array: ')
print(arr)
print("arr shape = ", arr.shape)
print(arr[:, np.newaxis])
print(arr[:, np.newaxis].shape)
print("-----------------------")

print("-----------------------")
print("yellow squares= ")
print(arr[2:4, 0:2])#.reshape(1, 4))
print("-----------------------")
# indices = [(0, 1), (1, 2), (2, 3), (3, 4)]
# print(', '.join(map(str, (arr[row, col] for row, col in indices))))

# print(arr[np.ix_([0, 4, 5], [3, 4])])
# print(arr[np.array([0, 4, 5])[:, None], [3, 4]])
print("-----------------------")
print(" Blue Squares")
print(arr[np.array([0, 4, 5])[:, np.newaxis], [3, 4]])
print("break:")
print(arr[np.array([0, 4, 5])])
print(arr[np.array([0])[:, np.newaxis]])
print(arr.shape)
# print(np.concatenate(np.array(arr[0][:, np.newaxis], arr[4][:, np.newaxis]))
print("-----------------------")
print("red squares= ", arr[[0, 1, 2, 3], [1, 2, 3, 4]])
print("-----------------------")




# Create a 1D array (row vector)
row_vector = np.array([0, 4, 5])

# Use np.newaxis to convert it to a column vector
column_vector = row_vector[:, np.newaxis]

print("Row Vector Shape:", row_vector.shape, row_vector)        # Output: (3,)
print("Column Vector Shape:", column_vector.shape, column_vector)  # Output: (3, 1)
print()

# arr2 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# print(arr2)
print(arr[[[0], [4], [5]], [3, 4]])
print()
print(np.array([0, 4, 5])[:, np.newaxis])
print()

X= np.array(
[[1,2,3],
[4,"5",6],
[7,8,9]
])
print(X.dtype)

print("------------------")
print("Satisfy condition:")
array_to_satisfy = np.array(range(0, 10))
array_to_satisfy[array_to_satisfy % 2 != 0] = -1
# print(np.mod(array_to_satisfy, 2))
print(array_to_satisfy)