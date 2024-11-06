"""
- ANN (Artificial Neural Networks):
    1. Divided in the Input Layer, Hidden Layers and the Output Layer
        -> If it has more than 1 hidden layer -> Deep Learning

    - Integration Function:
        The weighted average + the bias
    - Activation Function:
        Produces the output pf the neuron.  -> Linear, Sigmoid, Relu, elu, etc.

- Training a Neural Network: Split dataset in Training, Validation and Test data
    1. Training data: Used to train the model
        - Typically 80-90% of the data
        - All classes a represented equally
    2. Validation Data: Used during training to check on how well the training is doing
        - Checks whether there is overfitting or not
        - Typically 5-10% of the data
    3. Test data: Used after training to evaluate how good the model is doing/was trained
        - Typically 5-10% of the data

The loss function measures how good the model is performing (the lower, the better). The goal of the training is to minimize the loss function.

Backpropagation is adapting the weights and biases based on the output of the loss function.
This is done by an optimizer algorithm called gradient descent. Today a variant of this algorithm called adam is mostly used.

Backpropagation is done after every batch. So in our very simple example, the feedforward/backpropagation is executed 3 times when going ones over the training dataset (1 epoch).

Training is done for a number of epochs (1 epoch is 1 pass over the training dataset). After every epoch, the loss function is also calculated for the validation data set. The number of epochs used for the training is a hyperparameter you have to define. It is also possible to automatically stop the training if the loss function is not going down anymore

The learning rate is the step size in the gradient descent or adam optimizer
• A small learning rate will be slow and requires a lot of epochs
• A too large learning rate will not find the minimum of the loss function
• Typical learning rate are between 0.01 and 0.0001
• Sometimes a learning rate schedule is defined, starting with a bigger value and decreasing the step after a number of epochs

Overfitting is happening when the model is performing well on the training dataset, but performing worse on the validation dataset (validation loss is higher than training loss). Most of the time, this is coming from the fact that the dataset is too small to train the model. Sometime data augmentation techniques are used to create more samples (e.g. create more pictures by rotating the existing ones a bit)

A lot of hyperparameters need to be defined for the NN model
• Number of layers
• Number of nodes in the layers
• Number of epochs
• Batch size
• Learning rate
• Type of activation function
• Type of loss function
• …
In reality several model with different hyperparameters are training and the model giving the best results for the test dataset (as this one is not used during training) is chosen. This process is called hyperparameter tuning.

- One-Hot Encoding: transforms categorical variable into binary columns
    When the classification model predict a class, each output node will give the probability for the corresponding class. The sum of all the output nodes (sum of all the probabilities) is equal to 1. To enforce this, the softmax activation function is used for the output layer together with a categorical_crossentropy loss.

For one-hot encoding the to_categorical function from tensorflow.keras.utils can be used.
In case the dataset is a pandas dataframe, the pd.get_dummies function can be used.

- Remark: in case of a binary classification (“positive”/”negative”), it is also possible to have 1 output node (so not using one-hot encoding), giving the probability of “positive”. In this case a sigmoid activation is used (giving a value between 0 and 1) together with a binary_crossentropy loss function.

- How a neural network works and its steps/process:

    - An epoch represents a full cycle through the entire training dataset during neural network training. Let’s go step-by-step to understand the training process, how a neural network is built, and what validation and training scores signify.

     1. What is an Epoch?
            - An epoch means the model sees every data point in the training set once.
            - Each epoch involves breaking the data into smaller subsets called batches, which are used to update the model’s parameters.

    2. What Happens Within Each Epoch:
        - The model processes each batch, adjusts parameters through backpropagation, and updates weights to reduce the error (loss).
        - After an epoch, the model's overall performance is evaluated, and the next epoch begins with adjusted weights.

Structure of a Neural Network
    1. Layers in the Network:
        - Input Layer: The first layer receives the raw data. In the case of image data like MNIST, where images are 28x28 pixels, it might have 784 nodes (one for each pixel).
        - Hidden Layers: These layers transform the input data by applying various weights and activation functions. Common layers include Dense (fully connected) layers with an activation function, like ReLU, which helps the network capture complex relationships in data.
        - Output Layer: The final layer outputs predictions. For a 10-class classification (e.g., MNIST digits 0-9), it typically has 10 nodes with a softmax activation to output probabilities for each class.
    2. Activation Functions:
        - Each layer has an activation function (e.g., ReLU for hidden layers, softmax for output layers). Activations introduce non-linearity, enabling the network to learn from complex patterns.

Model Architecture and Training Process
    1. Forward Pass:
        - During each batch, the input data is passed forward through the network layers.
        - At each layer, the model applies weights, sums them with a bias term, and applies the activation function. This continues until the final output layer, where a prediction is made.

    2. Loss Calculation:
        - A loss function (e.g., categorical cross-entropy for multi-class classification) calculates the difference between the predicted values and the actual labels.

    3. Backward Pass (Backpropagation):
        - After calculating the loss, backpropagation occurs, where gradients (the rate of change in error) are calculated for each weight.
        - Using an optimizer like Adam, weights are adjusted to minimize the loss.

    4. Updating Weights:
        - The optimizer updates the weights based on the gradients, aiming to reduce the overall loss in the next forward pass.

    5. End of an Epoch:
        - The model has now seen the entire dataset once, completing an epoch.
        - After each epoch, metrics like training and validation loss and accuracy are logged.

What is the Validation Score?
    1. Purpose of Validation:
        - The validation score (accuracy or loss on the validation set) is used to evaluate how well the model generalizes to unseen data.
        - The validation set (distinct from the training set) helps to detect overfitting, where a model performs well on training data but poorly on new data.
    2. When and How it’s Used:
        - After each epoch, the model is tested on the validation set, and the validation loss/accuracy is recorded.
        - If validation performance degrades while training performance improves, this suggests overfitting.

    * Overfitting happens when training loss decreases continuously, but validation loss eventually starts to increase. This means the model is doing well on the training data (memorizing it) but performing worse on new, unseen data (not generalizing well).

    - In other words, during overfitting:
        * Training loss keeps getting lower (improving).
        * Validation loss initially decreases but then starts increasing, showing the model’s performance on new data is worsening.

    -> So, a true sign of overfitting is when the training performance continues to improve, but validation performance starts to degrade (increase in validation loss)

Training Score
    1. Training Score Calculation:
        - The training score (e.g., accuracy or loss) shows how well the model is learning from the training data.
        - A low training loss indicates that the model is fitting the training data well, while high accuracy means it’s making correct predictions frequently.

    2. How It’s Used:
        - The training score helps monitor progress during each epoch, but it must be balanced with the validation score to avoid overfitting.
        - Ideally, both training and validation scores improve together, indicating the model is learning useful features and generalizing well.

Summary of the Full Process
    1. Data goes through multiple epochs.
    2. In each epoch, batches are processed, and the loss is calculated and minimized using backpropagation.
    3. The model’s weights are updated each batch to reduce the error.
    4. After each epoch, the validation score checks if the model generalizes well.
    5. Training score tracks how well the model fits the training data, while validation score prevents overfitting.

Expected Outputs
    - Training Loss and Accuracy: Indicate how well the model fits training data.
    - Validation Loss and Accuracy: Reflect how well the model generalizes to new data.
    - Predictions: The model outputs probabilities per class (for each digit 0-9), from which the most probable class is chosen as the prediction.

This full training and evaluation cycle helps the model adjust to recognize patterns, make accurate predictions, and generalize well to new data.
"""

"""
- Steps To Follow:
    1. Figure out what data you're predicting (continuous or discrete)
        -> If discrete (categories/classes) then you're going to do classification
        -> If continuous (numerical...) then you're going to do regression
    2. Load data into DataFrame
        -> Load data as x
        -> Load target as y
    3. Clean data
        -> Remove rows with null values
    4. Normalize data (if the value are continuous and not categorical)
        -> MinMaxScaler to fit and transform the data
        -> By default most of the time the data will be continuous data so it will mostly always be necessary to normalize
        -> Only normalize the target data if its continuous data and not categorical
    5. Split the data
        -> Split data in training and test data
    6. Convert output in the right format
        -> To categorical with one hot encoding
    7. Define the model
        -> 
"""
















# Notebook 6.1:
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from plot_loss import plot_loss # own function in plot_loss.py


"""This code builds, trains, and evaluates a simple neural network for digit classification using the MNIST dataset, which is a set of 28x28 pixel grayscale images of handwritten digits (0-9). Here’s a step-by-step breakdown:

1.Importing Required Libraries
    - matplotlib.pyplot as plt: Used to display images and visualize the input data.
    - MinMaxScaler (from sklearn.preprocessing): Scales data to a range (0-1) to improve the model’s learning efficiency.
    - tensorflow.keras classes: These create and train the neural network, including loading the data, defining the model architecture, and training configurations.

2. Loading and Exploring the Data"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

"""
3. Reshaping Data (Images in this case)
    - Reshapes each 28x28 image into a 1D array of 784 values (28 * 28 = 784), flattening it for the neural network. This means each image is now represented as a 784-dimensional vector, suitable for feeding into a densely connected (fully connected) neural network.
    - x_train.shape confirms that the reshaped training data has 60,000 samples, each with 784 features."""
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

"""
4. Scaling the Input Data
    - The data values (0-255 for pixel brightness) are scaled to a range of 0 to 1 using MinMaxScaler. Normalizing helps the model learn faster and improves performance."""
scaler = MinMaxScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

"""
5. One-Hot Encoding the Labels
    - Converts labels into one-hot encoded format (i.e., each digit label is represented as a 10-element array with 1 in the position of the digit and 0s elsewhere). For example, the digit 5 becomes [0, 0, 0, 0, 0, 1, 0, 0, 0, 0].
    - This encoding is necessary for the categorical cross-entropy loss function used in the model."""
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

"""
6. Defining the Neural Network Architecture
    - The input layer takes 784 features (1 per pixel in the flattened images).
    - There are two hidden layers with 128 and 64 nodes, using the ReLU activation function to introduce non-linearity.
    - The output layer has 10 nodes, one per digit (0-9), with a softmax activation to produce probabilities for each class. The model structure, shown with model.summary(), will detail the layer connections and parameter counts."""
inputs = Input(shape=(784,))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs, name='MNIST')
print(model.summary())

"""
7. Compiling the Model
    - Uses the Adam optimizer for efficient gradient descent and a learning rate of 0.001.
    - Categorical cross-entropy is the loss function, suitable for multi-class classification tasks. The model will also report accuracy during training and evaluation.
"""
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

"""
8. Training the Model
    - Trains the model on the normalized training data (x_train_norm and y_train_onehot) for 5 epochs, with batches of 32 samples. The model holds out 10% of the training data for validation, helping to monitor overfitting during training."""
history = model.fit(
    x_train_norm,
    y_train_onehot,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
)
plot_loss(history)


"""
9. Evaluating the Model
    - Tests the model on the normalized test data to get a final evaluation of performance. This outputs the accuracy and loss on unseen data."""
print(model.evaluate(x_test_norm, y_test_onehot))


"""
10. Prediction with the Model
    - Runs predictions on the first two samples from the test set. predicted will contain probabilities for each class. To interpret the predicted class, we’d generally take the argmax (index of the maximum probability), indicating the model’s classification.
    - y_test[:2] shows the true labels of these first two test samples, allowing comparison with the model's predictions."""
predicted = model.predict(x_test_norm[:2])
# print("x_test_norm[:2]: ", x_test_norm[:2])
print(predicted)
# what it should be
print(y_test[:2])





f"""
Steps for neural networks for regression:
    {print("\n\nSteps for neural networks for regression:\n")}    
"""
# Imports:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from plot_loss import plot_loss # own function in plot_loss.py

# - Store data in dataframe
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv("/home/anir333/Desktop/KDG/Subjects/Year_2/Semester_1/DataAndAI/Python/dai2_dai3/Week6/06-Artificial neural networks/ dataset/auto-mpg.data", names=column_names, na_values='?', sep=' ', comment='\t', skipinitialspace=True)


# - Data Cleaning (Drop rows that have null values):
print(dataset.isna().sum())
dataset = dataset.dropna()
print(dataset.isna().sum())
print(dataset.head())


# - Set the features/target values:
x = dataset[["Cylinders","Displacement", "Horsepower", "Weight", "Acceleration", "Model Year"]]
y = dataset[['MPG']]

print(x.describe())

# - Normalize using MinMaxScaler():
scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x)

# - Split data into a training and test dataset:
x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.1)


# - Define the model:
inputs = Input(shape=(6,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)
model = Model(inputs, outputs, name='auto-mpg')
print(model.summary())


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_percentage_error']
)

# - Train the model:
# train the model
history = model.fit(
    x_train, # training input
    y_train, # training targets
    epochs=100,
    batch_size=32,
    validation_split=0.1,
)
# plot loss function
plot_loss(history)


# - Evaluate model with test data (look at MAE):
print(model.evaluate(x_test,y_test))


# - Predict:
y_predicted = model.predict(x_test)


# Plot predicted values against true value:
a = plt.axes(aspect='equal')
plt.scatter(y_test, y_predicted)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()


# Plot Error:
error = y_predicted - y_test
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
plt.ylabel('Count')
plt.show()










"""
- Formulas:
    1. Sigmoid
Sigmoid squashes the input to a value between 0 and 1, making it useful for binary classification
"""
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

"""
2. ReLU (Rectified Linear Unit)
ReLU returns 0 for negative inputs and the input itself for positive values, which introduces non-linearity and reduces computation complexity.
"""
# in Pictures

"""
- Lambda:
"""
import numpy as np

# Linear
linear = lambda z: z

# Sigmoid
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# ReLU
relu = lambda z: np.maximum(0, z)

# Leaky ReLU (using alpha=0.01)
leaky_relu = lambda z: np.where(z > 0, z, 0.01 * z)

# Tanh
tanh = lambda z: np.tanh(z)

# ELU (using alpha=1.0)
elu = lambda z: np.where(z >= 0, z, 1.0 * (np.exp(z) - 1))

# Softmax
softmax = lambda z: np.exp(z) / np.sum(np.exp(z), axis=0)


"""
- Why do we do each:
Let me break down why each of these steps is important in the context of preparing data for a machine learning model:

### 1. **Scale the input using Min-Max Scaling**
   **Why we do this:**
   - **Reason:** Features (input data) in machine learning can have vastly different ranges or units, for example, one feature might range from 0 to 1, while another might range from 1,000 to 10,000. If we don't scale these features, the model might give more importance to larger numbers simply because they have higher values, which could bias the learning process.
   - **Min-Max Scaling**: This scaling technique brings all the features into a common range (typically between 0 and 1). It ensures that no particular feature dominates the learning process just due to its larger scale.

   **Example:**
   If a feature like age ranges from 0 to 100, and another feature like income ranges from 1000 to 100,000, the model might give more importance to income. Min-max scaling ensures that both features contribute equally to the model’s learning process.

### 2. **Split the data into train (85%) and test (15%) sets**
   **Why we do this:**
   - **Reason:** To evaluate the performance of a model on data it hasn't seen during training, ensuring it generalizes well to new, unseen data (avoiding overfitting). The training set is used to train the model, and the test set is used to assess how well the model performs after being trained.
   - **Training set (85%)**: The model learns from this data. It adjusts its internal parameters (weights, biases) to minimize error.
   - **Test set (15%)**: This data is kept aside and is never used during training. After training, we test the model on this data to simulate how it would perform on new, real-world data.
   
   **Reason for 85%-15% split**: The typical ratio is 80-90% for training and 10-20% for testing. This split gives the model enough data to learn from (85%) while still allowing for meaningful evaluation (15%).

### 3. **Put the output in the right format (One-Hot Encoding)**
   **Why we do this:**
   - **Reason:** In classification problems, the output is usually categorical (e.g., 0, 1, 2, 3 for four different classes). Neural networks, however, typically perform best when the output labels are in a numerical format.
   - **One-Hot Encoding**: This technique transforms categorical labels (such as 0, 1, 2) into binary vectors. For example, if there are three classes, the label 0 becomes `[1, 0, 0]`, label 1 becomes `[0, 1, 0]`, and label 2 becomes `[0, 0, 1]`. This is necessary for training models like neural networks, which expect numeric output values for their final layer to calculate probabilities or classifications.
   
   **Example**:
   If you're classifying animals into categories like "dog", "cat", and "fish", you might use 0, 1, and 2 to represent each category. Using one-hot encoding transforms this into three-dimensional vectors: `[1, 0, 0]` for dog, `[0, 1, 0]` for cat, and `[0, 0, 1]` for fish.

### In Summary:
- **Min-Max Scaling**: Ensures all features are on a similar scale, preventing features with larger values from dominating the model's learning process.
- **Train-Test Split**: Helps ensure that the model is not overfitting and generalizes well to unseen data.
- **One-Hot Encoding**: Converts categorical output labels into a numerical format that is compatible with machine learning algorithms like neural networks.
"""


"""
- Why we do each summerized:

Here’s a simplified explanation of why we do each step:

1. **Min-Max Scaling**: 
   - **Why**: It makes sure that all input features (like age, salary) are on the same scale. Without this, some features might dominate the model just because their numbers are bigger.
   - **In simple terms**: We make sure everything is in the same "range" so the model can treat all features equally.

2. **Train-Test Split**:
   - **Why**: We need to train the model with some data, but also test it on data it has never seen before. This helps us understand how well it will work in real life (with new data).
   - **In simple terms**: We train the model on one set of data and test it on another to see if it can make accurate predictions.

3. **One-Hot Encoding**:
   - **Why**: Machine learning models need numerical data to work. One-hot encoding turns categories (like "dog", "cat", "fish") into numbers so the model can process them correctly.
   - **In simple terms**: We change categories into numbers so the model can understand them.

Each of these steps is about preparing the data in a way that helps the model learn more effectively and make accurate predictions.
"""

"""
What type to use and when: classification/regression:
When deciding between a neural network classification model and a regression model, the choice depends on the nature of your output variable (the target you want the model to predict). Here’s how to know which to use:

1. **Classification**: Use a neural network for classification if you’re predicting **categories or classes**.
   - **Output**: The target variable is categorical (e.g., classes like "dog," "cat," or "fish," or binary outcomes like 0 or 1 for "spam" vs. "not spam").
   - **Examples**:
     - Image classification (e.g., identifying digits in MNIST, distinguishing between types of animals).
     - Sentiment analysis (positive, negative, or neutral).
     - Predicting if a customer will buy a product (yes/no).

2. **Regression**: Use a neural network for regression if you’re predicting a **continuous value**.
   - **Output**: The target variable is numerical and continuous (e.g., predicting a score, price, or any real number).
   - **Examples**:
     - Predicting house prices based on features like square footage, location, etc.
     - Forecasting stock prices.
     - Estimating temperature over time.

### Key Indicators of Which to Use:
- **Ask yourself what type of answer you need**: If the answer is a specific category, use classification. If it's a number, use regression.
- **Check the target variable**: 
  - If it’s a list of categories or labels (discrete values), that’s a good indicator for classification.
  - If it’s a range of continuous numbers, that indicates regression.
  
### When You Might Be Told:
In many cases, especially in real-world applications, you’ll know the task beforehand. For example, if the problem is presented as “predicting a category” or “predicting a number,” this is your cue. However, if you’re exploring a dataset yourself, examining the type of target variable is the best guide to decide.


"""