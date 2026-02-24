import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class WaterQualityDataOrganization:

    training_data: pd.DataFrame
    testing_data: pd.DataFrame

    x_train: np.ndarray
    x_test: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_val: np.ndarray

    #Xtrain_3d_split: list
    #Xtest_3d_split: list
    #Xval_3d_split: list

    x_train_3d: np.ndarray
    x_test_3d: np.ndarray
    x_val_3d: np.ndarray

    def __init__(self,data):
        self.training_data, self.testing_data = data

    def preprocessing(self):
        Xtrain = self.training_data.iloc[:,1:-1]
        Ytrain = np.array(self.training_data['EVENT'])

        Xtest = self.testing_data.iloc[:,1:-1]
        Ytest = np.array(self.testing_data['EVENT'])

        Xtrain = Xtrain.ffill()
        Xtest = Xtest.ffill()
        #fill the empty spaces

        window_trend = 24 * 60

        Xtrain_trend = Xtrain.rolling(window_trend, min_periods=1).mean()
        Xtest_trend = Xtest.rolling(window_trend, min_periods=1).mean()

        Xtrain = Xtrain - Xtrain_trend
        Xtest = Xtest - Xtest_trend

        # separation on validation and training
        Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2)

        # normalziation
        means = Xtrain.mean()
        stds = Xtrain.std()

        Xtrain = (Xtrain - means) / stds
        Xval = (Xval - means) / stds
        Xtest = (Xtest - means) / stds
        self.x_val = Xval
        self.x_train = Xtrain
        self.x_test = Xtest
        self.y_val = Yval
        self.y_train = Ytrain
        self.y_test = Ytest


    def shape_to_3d(self,window):

        Xtrain = self.x_train
        Xval = self.x_val
        Xtest = self.x_test

        window = window

        Xtrain_ext = pd.concat([pd.DataFrame([Xtrain.iloc[0]]*(window-1)), Xtrain])
        Xval_ext = pd.concat([pd.DataFrame([Xval.iloc[0]]*(window-1)), Xval])
        Xtest_ext = pd.concat([pd.DataFrame([Xtest.iloc[0]]*(window-1)), Xtest])

        Xtrain_3d = np.zeros((Xtrain.shape[0], window, Xtrain.shape[1]))
        Xval_3d = np.zeros((Xval.shape[0], window, Xval.shape[1]))
        Xtest_3d = np.zeros((Xtest.shape[0], window, Xtest.shape[1]))

        for i in tqdm(range(Xtrain.shape[0])):
            data = Xtrain_ext.iloc[i:i+window,:]
            # Xtrain_3d[i] = (data - data.mean()) / data.std()
            Xtrain_3d[i] = (data - data.mean())
            # Xtrain_3d[i] = Xtrain_ext.iloc[i:i+window,:]
        for i in tqdm(range(Xval.shape[0])):
            data = Xval_ext.iloc[i:i+window,:]
            # Xval_3d[i] = (data - data.mean()) / data.std()
            Xval_3d[i] = (data - data.mean())
            # Xval_3d[i] = Xval_ext.iloc[i:i+window,:]
        for i in tqdm(range(Xtest.shape[0])):
            data = Xtest_ext.iloc[i:i+window,:]
            # Xtest_3d[i] = (data - data.mean()) / data.std()
            Xtest_3d[i] = (data - data.mean())
            # Xtest_3d[i] = Xtest_ext.iloc[i:i+window,:]

        self.x_train_3d = Xtrain_3d
        self.x_val_3d = Xval_3d
        self.x_test_3d = Xtest_3d
        #n_features = Xtrain_3d.shape[2]

        #self.Xtrain_3d_split = [Xtrain_3d[:, :, i].reshape(Xtrain_3d.shape[0], Xtrain_3d.shape[1], 1) for i in range(n_features)]
        #self.Xval_3d_split = [Xval_3d[:, :, i].reshape(Xval_3d.shape[0], Xval_3d.shape[1], 1) for i in range(n_features)]
        #self.Xtest_3d_split = [Xtest_3d[:, :, i].reshape(Xtest_3d.shape[0], Xtest_3d.shape[1], 1) for i in range(n_features)]
