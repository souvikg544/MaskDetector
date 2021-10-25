import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model=load_model("maskmodel4")
#detector = MTCNN()
cn=['with mask', 'Without mask']
font = cv2.FONT_HERSHEY_SIMPLEX

video_capture = cv2.VideoCapture('testvideo.3gpp')

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
    arr=[]
    loc=[]

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        arr.append(img_to_array(face))
        locs.append()

    prediction = model.predict(np.expand_dims(arr, 0))
    confidnce = round(100 * (np.max(prediction[0])), 2)
    predicted_class = cn[np.argmax(prediction[0])]
    if (predicted_class == 'with mask' and confidnce <= 97.0):
        predicted_class = "Without Mask"

    cv2.putText(frame,
                predicted_class + str(confidnce),
                (50, 50),
                font, 0.7,
                (0, 255, 255),
                2,
                cv2.LINE_4)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()