from Data_preparation import dataPreparation
from Model import learningModel
from keras import backend as K
import tensorflow as tf

X_train,Y_train,X_test,Y_test = dataPreparation()
print(X_train)
model = learningModel( X_train,Y_train,X_test,Y_test)
print(model.evaluate(X_test,Y_test))
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("model saved")