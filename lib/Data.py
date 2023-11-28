# Python 3.10.1
# Aditya Sawant
import scipy.io as sio
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import json
import os
try:
    from lib.Metadata import metadata
except:
    from Metadata import metadata
import h5py
class Data:
    
    def __init__(self, dataset, pca_components=30, windowSize=11):
        self.dataset = dataset
        self.X: np.ndarray
        self.Y: np.ndarray
        self.X_pca: np.ndarray 
        self.Y_og: np.ndarray
        self.datasetShape: tuple
        self.patches: list
        self.load_json()
        self.load_data()
        self.apply_pca(pca_components)
        self.createImageCubes(windowSize)
        self.classWisePatches()
        
        
    def load_json(self):
        self.dataset_meta = metadata[self.dataset]
        print(self.dataset_meta['name'])
        return self.dataset_meta
        
    def load_data(self):
       X, Y = self.get_original_data()
       self.Y_og = Y
       self.X = X
       self.Y = Y
       self.datasetShape = self.X.shape
       return self.X, self.Y
            
    
    def get_original_data(self):
        try:
            X = sio.loadmat(f'{os.getcwd()}\\Datasets\\{self.dataset_meta["name"]}{self.dataset_meta["data"]["suffix"]}{self.dataset_meta["data"]["ext"]}')[ self.dataset_meta["data"]["key"] ]
            Y = sio.loadmat(f'{os.getcwd()}\\Datasets\\{self.dataset_meta["name"]}{self.dataset_meta["label"]["suffix"]}{self.dataset_meta["label"]["ext"]}')[ self.dataset_meta["label"]["key"] ]
            return X, Y
        except Exception as e:
            data = h5py.File(f'{os.getcwd()}\\Datasets\\{self.dataset_meta["name"]}{self.dataset_meta["data"]["suffix"]}{self.dataset_meta["data"]["ext"]}', 'r')
            label = h5py.File(f'{os.getcwd()}\\Datasets\\{self.dataset_meta["name"]}{self.dataset_meta["label"]["suffix"]}{self.dataset_meta["label"]["ext"]}', 'r')
            X = data[self.dataset_meta["data"]["key"]]
            Y = label[self.dataset_meta["label"]["key"]]
            X = np.transpose(np.array(X), axes=[2, 1, 0])
            Y = np.transpose(np.array(Y), axes=[1, 0])
            return X, Y
    
    def apply_pca(self, n_components):
        pca = PCA(n_components=n_components, whiten=True) # create a PCA object
        new_X = np.reshape(self.X, (-1, self.X.shape[2])) # reshape the data into a 2D matrix
        new_X = pca.fit_transform(new_X) # fit the PCA object
        new_X = np.reshape(new_X, (self.X.shape[0], self.X.shape[1], n_components)) # reshape the data into a 3D matrix
        self.X = new_X
        self.X_pca = new_X
        del pca, new_X# delete the PCA object
        return self.X_pca
    
    def padWithZeros(self, X, margin):
        return np.pad(X, ((margin,margin), (margin,margin), (0,0)), 'constant', constant_values=0)
    
    def createImageCubes(self, windowSize):
        margin = int(windowSize // 2)
        zeroPaddedX = self.padWithZeros(self.X, margin)
        dataPatches = [zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1] for r in range(margin, zeroPaddedX.shape[0] - margin) for c in range(margin, zeroPaddedX.shape[1] - margin)]
        dataPatches = np.expand_dims(dataPatches, axis=-1)
        dataLabels = [self.Y[r-margin, c-margin] for r in range(margin, zeroPaddedX.shape[0] - margin) for c in range(margin, zeroPaddedX.shape[1] - margin)]
        self.X = dataPatches
        self.Y = np.array(dataLabels)
        return self.X, self.Y
        
        
    def createPatches(self, windowSize):
        margin = int(windowSize // 2)
        zeroPaddedX = self.padWithZeros(self.X_pca, margin)
        dataPatches = np.ndarray(
            shape=(self.X_pca.shape[0], self.X_pca.shape[1], windowSize, windowSize, self.X_pca.shape[-1]))
        print(dataPatches.shape)
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            for r in range(margin, zeroPaddedX.shape[0] - margin):
                dataPatches[r - margin, c - margin] = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
        return dataPatches, self.Y_og

    def classWisePatches(self):
        patches =  [ self.X[self.Y==i,:,:,:,:] for i in range(1,self.dataset_meta['num_classes']+1) ]
        self.patches = patches
        return self.patches
    
    def get_data(self):
        return self.X, self.Y, self.patches
    
    def get_pca_data(self):
        return self.X_pca, self.Y
        
    def get_dataset_shape(self):
        return self.datasetShape
    
    def get_num_classes(self):
        return self.dataset_meta['num_classes']
    def load_defaults(self):
        NUM_CLASSES = self.dataset_meta['num_classes']
        TRAINING_CLASSES = self.dataset_meta['training_classes']
        TRAINING_LABELS: list = list(map(lambda x: x+1, TRAINING_CLASSES)) # Labels to be used for training
        TESTING_CLASSES = self.dataset_meta['testing_classes']
        TESTING_LABELS: list = list(map(lambda x: x+1, TESTING_CLASSES))
        TUNNING_LABELS = TESTING_LABELS
        
        TRAINING_PATCHES: list = [self.patches[i] for i in TRAINING_CLASSES]
        TESTING_PATCHES: list = [self.patches[i] for i in TESTING_CLASSES]
        print(len(TESTING_PATCHES))
        TUNNING_PATCHES:list = [TESTING_PATCHES[i][:5,:,:,:,:] for i in range(len(TESTING_CLASSES))]
        
        return NUM_CLASSES, TRAINING_CLASSES, TRAINING_LABELS, TUNNING_LABELS, TESTING_CLASSES, TESTING_LABELS, TRAINING_PATCHES,TUNNING_PATCHES, TESTING_PATCHES
    
    def get_target_names(self):
        return self.dataset_meta['target_names']

if __name__ == '__main__':
    d = Data('IP', 30, 1)
    _X, _Y = d.get_original_data()
    print(_X.shape, _Y.shape)
    X, Y, patches = d.get_data()
    _X_pca, _Y = d.get_pca_data()
    print(_X_pca.shape, _Y.shape)
    print(X.shape, Y.shape)