# Python 3.10.1
# Aditya Sawant
try:
    from Data import Data
    from PrototypicalNetwork import Prototypical
except:
    from lib.Data import Data
    from lib.PrototypicalNetwork import Prototypical
import tensorflow as tf
import numpy as np
import random
from tqdm.auto import tqdm


def generateWindow(X: np.ndarray, Y: np.ndarray, patches: list, windowSize: int, imageData: tuple, C, K):

    (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL) = imageData
    
    for y in range(0, X.shape[1], windowSize ):
        for x in range(0, X.shape[0], windowSize):
            
            queryPatches = X[x:x+windowSize, y:y+windowSize, :, :, :]
            queryLabels = Y[x:x+windowSize, y:y+windowSize]
            supportPatches = []
            supportLabels = [i for i in range(1, C+1)]
            for n in supportLabels:
                sran_indices = np.random.choice(len(patches[n-1]),K,replace=False)  # for class no X-1: select K samples 
                supportPatches.extend( patches[n-1][sran_indices,:,:,:,:])
            supportPatches = tf.convert_to_tensor(np.reshape(np.asarray(supportPatches),(C*K,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL)),dtype=tf.float32)
    

            w = len(queryLabels)
            h = len(queryLabels[0])
            queryLabels = np.reshape(np.asarray(queryLabels), (len(queryLabels) * len(queryLabels[0])))
            queryPatches = tf.convert_to_tensor(np.reshape(np.asarray(queryPatches),(len(queryLabels),IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL)),dtype=tf.float32)
            yield x, y, queryLabels.tolist(), queryPatches, supportLabels, supportPatches ,w, h

# 0 1 2 3 4 5 6 7 8 9 10 11



def predictImage(Data: Data, ProtoModel: Prototypical , imageData: tuple, N_TIMES, windowSize):
    (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL) = imageData
    X, Y = Data.createPatches()
    _, _, X_patchwise = Data.get_data()
    C = Data.get_num_classes()
    X_shape = Data.get_dataset_shape()
    predictions = np.zeros((X_shape[0], X_shape[1]))
    K = 5
    all_preds = []
    all_y_preds = []
    # print(predictions.shape)
            
    

    epochObj = tqdm(generateWindow(X, Y, X_patchwise, windowSize, imageData, C, K), desc=f'Patch', total=((X.shape[0]//windowSize) * (X.shape[1] //windowSize)))
    for x, y, queryLabels, queryPatches, supportLabels, supportPatches, w, h in epochObj:
        # print(x, y, len(queryLabels), queryPatches.shape, len(supportLabels), supportPatches.shape)
        loss, mean_predictions, mean_accuracy, classwise_mean_acc, y_preds = ProtoModel(supportPatches, queryPatches, supportLabels, queryLabels, K, C, len(queryLabels), training=False)
        correctIndices = tf.cast(tf.argmax(mean_predictions, axis=-1) + 1, tf.int32) 
        predictions[x:x+w,  y:y+h] = np.reshape(correctIndices.numpy(), (w, h))
        all_preds.extend(mean_predictions)
        all_y_preds.extend(y_preds)
    
    return predictions, Y, all_preds, all_y_preds
        


if __name__ == '__main__':
    Data = Data('IP')
    predictImage(Data, None, (11, 11, 30, 1))
