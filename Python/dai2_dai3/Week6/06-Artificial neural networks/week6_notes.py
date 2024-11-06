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
"""
# # Notebook 6.1:
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
print(predicted)
# what it should be
print(y_test[:2])