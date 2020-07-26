import pandas as pd
import numpy as np
from keras.utils import np_utils

def dataPreparation():
    emotion_data = pd.read_csv('./Data/fer2013.csv')
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    for row in emotion_data.iterrows():
        k = row['pixels'].split(" ")
        
    
        if row['Usage'] =='Training':
            X_train.append(np.array(k))
            Y_train.append(row['emotion'])
        elif row['Usage'] =='PublicTest':
            X_test.append(np.array(k))
            Y_test.append(row['emotion'])
    

    X_train = np.array(X_train,dtype=np.float32)
    X_test  = np.array(X_test,dtype=np.float32)
    Y_test = np.array(Y_test,dtype=np.float32)
    Y_train = np.array(Y_train,dtype=np.float32)

    X_train = X_train.reshape(X_train.shape[0],48,48,1)
    X_test = X_test.reshape(X_test.shape[0],48,48,1)
    Y_train = np_utils.to_categorical(Y_train,num_classes=7)
    Y_test = np_utils.to_categorical(Y_test,num_classes=7)

    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test = dataPreparation()
print(X_test)