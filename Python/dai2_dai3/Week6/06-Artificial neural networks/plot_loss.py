import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    # print("\n\nHistory:\n", history)
    print("\n\nHistory.history:\n", history.history)
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'])
        plt.legend(['Train', 'Val'], loc='upper left')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
