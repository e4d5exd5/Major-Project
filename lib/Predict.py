# Python 3.10.1
# Aditya Sawant

from Data import Data
from PrototypicalNetwork import Prototypical
import tensorflow as tf
import numpy as np


def predictImage(Data: Data, ProtoModel: Prototypical or None, windowSize: int = 11):

    X, Y = Data.createPatches(windowSize)
    print(X.shape, Y.shape)
    X_shape = Data.get_dataset_shape()
    predictions = np.zeros((X_shape[0], X_shape[1]))
    print(predictions.shape)


if __name__ == '__main__':
    Data = Data('IP')
    predictImage(Data, None, 11)
