from keras.models import Sequential
from keras.layers.core import Flatten,Dense,Dropout
from keras.layers.convolutional import Conv2D,ZeroPadding2D,MaxPooling2D
from keras.optimizers import SGD

def learningModel(X_train,Y_train,X_test,Y_test):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3),activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3),activation='relu'))
    model.add(MaxPooling2D((2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(2048,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7,activation='softmax'))

    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(X_train,Y_train,batch_size=32,epochs=30,verbose=1,validation_data=(X_test,Y_test))

    return model


