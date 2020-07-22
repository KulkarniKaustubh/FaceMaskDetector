import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as K

fileJSON = open('mask-detect-model.json', 'r')
jsonModel = fileJSON.read()
fileJSON.close()
model = K.models.model_from_json(jsonModel)
model.load_weights('mask-detect-model-weights.h5')

# model.summary()

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def drawRectangle(image, confidence):
    points = faceCascade.detectMultiScale(img)

    for (x, y, w, h) in points:
        if confidence > 0.5:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)  # BGR
        elif confidence <= 0.5:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

feed = cv2.VideoCapture(0)

while True:
    ret, img = feed.read()

    imgCopy = img.copy()
    imgCopy = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2RGB)
    imgCopy = cv2.resize(imgCopy, (300, 300)) # trained on (300, 300)
    imgCopy = imgCopy/255.   # normalizing
    imgCopy = np.expand_dims(imgCopy, axis = 0) # this is done since the model is trained on mini batches, it has to have a 4D shape
    imgCopyTensor = tf.convert_to_tensor(imgCopy) # this is done to pass image as a tensor input to the model

    prediction = model.predict(imgCopyTensor, steps = 1) # only 1 frame at a time

    # print(prediction)

    drawRectangle(img, prediction)

    cv2.imshow('Mask Detect', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

feed.release()
cv2.destroyAllWindows()
