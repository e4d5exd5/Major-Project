import matplotlib.pyplot as plt
from time import time

def timeIt(func):
    """
    timeIt is a decorator function to time the execution of a function.

    :param func: function to be timed
    :return: wrapper function
    """
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


def plotData(data, testing=False):

    if testing:
        accuracy1_values = [item[0] for item in data]
        accuracy2_values = [item[1] for item in data]
        loss1_values = [item[2] for item in data]
        loss2_values = [item[3] for item in data]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss1_values, color='red', label='Loss')
        plt.plot(accuracy1_values, color='blue', label='Accuracy')
        plt.xlabel('Epoch/Episode')
        plt.ylabel('Value')
        plt.title('Loss and Overall Accuracy 1')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(loss2_values, color='red', label='Loss')
        plt.plot(accuracy2_values, color='blue', label='Accuracy')
        plt.xlabel('Epoch/Episode')
        plt.ylabel('Value')
        plt.title('Loss and Overall Accuracy 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        loss_values = [item[0] for item in data]
        accuracy_values = [item[1] for item in data]

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, color='red', label='Loss')
        plt.plot(accuracy_values, color='blue', label='Accuracy')
        plt.xlabel('Epoch/Episode')
        plt.ylabel('Value')
        plt.title('Loss and Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
