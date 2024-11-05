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
"""