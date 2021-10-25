import cv2
import sys
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model=load_model("maskmodel4")
#detector = MTCNN()
cn=['with mask', 'Without mask']
font = cv2.FONT_HERSHEY_SIMPLEX


video_capture = cv2.VideoCapture('testvideo.3gpp')
#frame = cv2.imread('hue.jpg',1)


def make720p():
    video_capture.set(3,1280)
    video_capture.set(4,720)

#make720p()


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # result = detector.detect_faces(frame)
    # if result != [] :
    #     for person in result:
    #         bounding_box = person['box']
    #         keypoints = person['keypoints']
    #
    #         cv2.rectangle(frame,
    #                       (bounding_box[0], bounding_box[1]),
    #                       (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
    #                       (0, 155, 255),
    #                       2)
    #         face = frame[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
    #         arr = img_to_array(face)
    #         prediction = model.predict(np.expand_dims(arr, 0))
    #         confidnce = round(100 * (np.max(prediction[0])), 2)
    #         predicted_class = cn[np.argmax(prediction[0])]
    #         # if (predicted_class == 'with mask' and confidnce <= 98.0 and confidnce >= 89):
    #         #     predicted_class = "Mask worn Incorrectly"
    #         #     confidnce = round(100 * prediction[0][0], 2)
    #         # elif (predicted_class == 'with mask' and confidnce < 89):
    #         #     predicted_class = "Without Mask"
    #         #     confidnce = round(100 * prediction[0][2], 2)
    #
    #         cv2.putText(frame,
    #                     predicted_class + str(confidnce),
    #                     (bounding_box[0], bounding_box[1]-40),
    #                     font, 0.5,
    #                     (255, 0, 0),
    #                     2,
    #                     cv2.LINE_4)

    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        arr=img_to_array(face)
        prediction=model.predict(np.expand_dims(arr,0))
        confidnce=round(100 * (np.max(prediction[0])), 2)
        predicted_class=cn[np.argmax(prediction[0])]
        if(predicted_class=='with mask' and confidnce <= 97.0):
            predicted_class="Without Mask"

        # elif(predicted_class == 'with mask' and confidnce < 89):
        #     predicted_class = "Without Mask"
        #     confidnce = round(100 * prediction[0][2], 2)

        cv2.putText(frame,
                    predicted_class + str(confidnce),
                    (x, y-50),
                    font, 0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)

    # i=1
    # if(i==1):
    #     cv2.imshow(face)
    #     i=i+1





    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()