import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image


model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')
print(cv2.__version__)
face_haar_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vs = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
	# read the next frame from the file
    (grabbed, frame) = vs.read()
 
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(frame)
    image_pixels = None
    
    for (x,y,w,h) in faces_detected:
          cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0))
          roi_gray=frame[y:y+w,x:x+h]
          roi_gray=cv2.resize(roi_gray,(48,48))
          cv2.rectangle(output,(x,y), (x+w,y+h), (255,0,0))

          image_pixels = np.array(roi_gray,dtype=np.float32)
          image_pixels = np.expand_dims(image_pixels, axis = 0)
          
          image_pixels = image_pixels.reshape((-1,48,48,1))

    if faces_detected!=():
        predictions = model.predict(image_pixels,steps=1)
        max_index = np.argmax(predictions[0])
        print(predictions[0])
        print(max_index)
        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]
        print(emotion_prediction)
        cv2.putText(output, emotion_prediction, (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        
    else:
        print("No Face detected")

    resized_img = cv2.resize(output, (800, 600))
    cv2.imshow('Emotion',resized_img)
    if cv2.waitKey(10) == ord('b'):
        break
vs.release()
cv2.destroyAllWindows