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
from scipy import stats
from tqdm.auto import tqdm

def generateSupportSet(patches: list, imageData: tuple, C, K):
    
    (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNEL) = imageData
    
    supportPatches = []
    supportLabels = [i for i in range(1, C+1)]
    for n in supportLabels:
        sran_indices = np.random.choice(len(patches[n-1]),K,replace=False)  # for class no X-1: select K samples 
        supportPatches.extend( patches[n-1][sran_indices,:,:,:,:])
    supportPatches = tf.convert_to_tensor(np.reshape(np.asarray(supportPatches),(C*K,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL)),dtype=tf.float32)
    
    
    return supportPatches, supportLabels


def generateWindow(X: np.ndarray, Y: np.ndarray, patches: list, windowSize: int, imageData: tuple, C, K):

    (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL) = imageData
    
    for y in range(0, X.shape[1], windowSize ):
        for x in range(0, X.shape[0], windowSize):
            
            queryPatches = X[x:x+windowSize, y:y+windowSize, :, :, :]
            queryLabels = Y[x:x+windowSize, y:y+windowSize]

            w = len(queryLabels)
            h = len(queryLabels[0])
            queryLabels = np.reshape(np.asarray(queryLabels), (len(queryLabels) * len(queryLabels[0])))
            queryPatches = tf.convert_to_tensor(np.reshape(np.asarray(queryPatches),(len(queryLabels),IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL)),dtype=tf.float32)
            yield x, y, queryLabels.tolist(), queryPatches, w, h

# 0 1 2 3 4 5 6 7 8 9 10 11



def predictImage(Data: Data, ProtoModel: Prototypical , imageData: tuple, N_TIMES, windowSize, VotingTimes=5, RandomSupport=True):
    (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,IMAGE_CHANNEL) = imageData
    X, Y = Data.createPatches()
    _, _, X_patchwise = Data.get_data()
    C = Data.get_num_classes()
    X_shape = Data.get_dataset_shape()
    predictions = np.zeros((X_shape[0], X_shape[1]))
    predictionsTrue = np.zeros((X_shape[0], X_shape[1]))
    K = 5

    # print(predictions.shape)
            
    

    epochObj = tqdm(generateWindow(X, Y, X_patchwise, windowSize, imageData, C, K), desc=f'Patch', total=((X.shape[0]//windowSize) * (X.shape[1] //windowSize)))
    for x, y, queryLabels, queryPatches, w, h in epochObj:
        # print(x, y, len(queryLabels), queryPatches.shape, len(supportLabels), supportPatches.shape)
        supportPatches, supportLabels = generateSupportSet(X_patchwise, imageData, C, K)
        
        votes = []
        
        for i in range(VotingTimes):
            loss, mean_predictions, mean_accuracy, classwise_mean_acc, y_preds = ProtoModel(supportPatches, queryPatches, supportLabels, queryLabels, K, C, len(queryLabels), training=False)
            correctIndices = tf.cast(tf.argmax(mean_predictions, axis=-1) + 1, tf.int32) 
            votes.append(correctIndices)
            if(RandomSupport):
                supportPatches, supportLabels = generateSupportSet(X_patchwise, imageData, C, K)
        
        # Calulate Majority Voting Here
        votes_tensor = tf.stack(votes, axis=-1)  # Stack the votes along the last axis
        majority_classes, _ = stats.mode(votes_tensor, axis=-1)
        # Fill in the predictions with the majority voting result
        predictions[x:x+w, y:y+h] = np.reshape(majority_classes, (w, h))
       
    
    y_test = []
    y_pred = []

    for x in range(X_shape[0]):
        for y in range(X_shape[1]):
            if(Y[x, y] != 0):
                y_test.append(predictions[x, y])
                y_pred.append(Y[x, y])
                predictionsTrue[x, y] = predictions[x, y]
    
    return predictions, predictionsTrue, Y, y_test, y_pred
        


if __name__ == '__main__':
    Data = Data('IP')
    predictImage(Data, None, (11, 11, 30, 1))