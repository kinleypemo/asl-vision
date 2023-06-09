from flask import Flask, render_template, Response, redirect, url_for
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from keras.models import load_model
import numpy as np
import math
import os
import tensorflow as tf

app = Flask(__name__)
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = tf.keras.models.load_model('Model\keras_model.h5')
model.save("model")
classifier = Classifier("model", "Model\labels.txt")

offset = 20
imgSize = 224

folder = "Data/C"
counter = 0

labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]


# def gen_frames():
#     while True:
#         success, img = cap.read()
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)
#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand["bbox"]

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

#             imgCropShape = imgCrop.shape

#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap : wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap : hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)

#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset - 50),
#                 (x - offset + 90, y - offset - 50 + 50),
#                 (255, 0, 255),
#                 cv2.FILLED,
#             )
#             cv2.putText(
#                 imgOutput,
#                 labels[index],
#                 (x, y - 26),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1.7,
#                 (255, 255, 255),
#                 2,
#             )
#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset),
#                 (x + w + offset, y + h + offset),
#                 (255, 0, 255),
#                 4,
#             )

#             cv2.imshow("ImageCrop", imgCrop)
#             cv2.imshow("ImageWhite", imgWhite)

#         cv2.imshow("Image", imgOutput)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
            
def gen_frames():
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > 0:
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        imgResizeShape = imgResize.shape
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap : wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                        print(prediction, index)

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > 0:
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        imgResizeShape = imgResize.shape
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap : hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset - 50),
                    (x - offset + 90, y - offset - 50 + 50),
                    (255, 0, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    imgOutput,
                    labels[index],
                    (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1.7,
                    (255, 255, 255),
                    2,
                )
                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset),
                    (x + w + offset, y + h + offset),
                    (255, 0, 255),
                    4,
                )

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", imgOutput)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def home():
   return render_template('home.html')
   
@app.route('/index')
def index():
   return render_template('index.html')
 
@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=5000 ,debug=True)


# from flask import Flask, render_template, Response, redirect, url_for
# import cv2
# import threading
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import tensorflow as tf

# app = Flask(__name__)
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# model = tf.keras.models.load_model('Model/keras_model.h5')
# model.save("model")
# classifier = Classifier("model", "Model/labels.txt")

# offset = 20
# imgSize = 224

# folder = "Data/C"
# counter = 0

# labels = [
#     "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
#     "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
# ]
# stop_event = threading.Event()

# def video_processing():
#     global cap
#     while True:
#         success, img = cap.read()
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)
#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand["bbox"]

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

#             imgCropShape = imgCrop.shape

#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap: wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap: hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)

#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset - 50),
#                 (x - offset + 90, y - offset - 50 + 50),
#                 (255, 0, 255),
#                 cv2.FILLED,
#             )
#             cv2.putText(
#                 imgOutput,
#                 labels[index],
#                 (x, y - 26),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1.7,
#                 (255, 255, 255),
#                 2,
#             )
#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset),
#                 (x + w + offset, y + h + offset),
#                 (255, 0, 255),
#                 4,
#             )

#             cv2.imshow("ImageCrop", imgCrop)
#             cv2.imshow("ImageWhite", imgWhite)

#         cv2.imshow("Image", imgOutput)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     stop_event.set()


# def gen_frames():
#     while not stop_event.is_set():
#         success, img = cap.read()
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)
#         if hands:
#             # ... your hand detection and processing code ...
#             hand = hands[0]
#             x, y, w, h = hand["bbox"]

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

#             imgCropShape = imgCrop.shape

#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap : wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap : hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)

#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset - 50),
#                 (x - offset + 90, y - offset - 50 + 50),
#                 (255, 0, 255),
#                 cv2.FILLED,
#             )
#             cv2.putText(
#                 imgOutput,
#                 labels[index],
#                 (x, y - 26),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1.7,
#                 (255, 255, 255),
#                 2,
#             )
#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset),
#                 (x + w + offset, y + h + offset),
#                 (255, 0, 255),
#                 4,
#             )

#             cv2.imshow("ImageCrop", imgCrop)
#             cv2.imshow("ImageWhite", imgWhite)

#         cv2.imshow("Image", imgOutput)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             stop_event.set()
#             break

#         ret, buffer = cv2.imencode('.jpg', img)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/index')
# def index():
#     return render_template('index.html')


# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/start_video_processing')
# def start_video_processing():
#     global stop_event
#     stop_event = threading.Event()
#     video_thread = threading.Thread(target=video_processing)
#     video_thread.start()
#     return redirect(url_for('video_feed'))


# @app.route('/stop_video_processing')
# def stop_video_processing():
#     stop_event.set()
#     return redirect(url_for('index'))


# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, render_template, Response, redirect, url_for
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import os
# import threading
# import tensorflow as tf

# app = Flask(__name__)
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# model = tf.keras.models.load_model('Model/keras_model.h5')
# model.save("model")
# classifier = Classifier("model", "Model/labels.txt")

# offset = 20
# imgSize = 224

# folder = "Data/C"
# counter = 0

# labels = [
#     "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
# ]

# stop_event = threading.Event()  # Event to signal when to stop the video processing
# video_thread = None  # Variable to hold the video processing thread


# def gen_frames():
#     while True:
#         success, img = cap.read()
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)
#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand["bbox"]

#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

#             imgCropShape = imgCrop.shape

#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap: wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)

#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap: hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)

#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset - 50),
#                 (x - offset + 90, y - offset - 50 + 50),
#                 (255, 0, 255),
#                 cv2.FILLED,
#             )
#             cv2.putText(
#                 imgOutput,
#                 labels[index],
#                 (x, y - 26),
#                 cv2.FONT_HERSHEY_COMPLEX,
#                 1.7,
#                 (255, 255, 255),
#                 2,
#             )
#             cv2.rectangle(
#                 imgOutput,
#                 (x - offset, y - offset),
#                 (x + w + offset, y + h + offset),
#                 (255, 0, 255),
#                 4,
#             )

#             cv2.imshow("ImageCrop", imgCrop)
#             cv2.imshow("ImageWhite", imgWhite)

#         cv2.imshow("Image", imgOutput)

#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     stop_event.set()  # Signal the video processing to stop


# def video_processing():
#     for frame in gen_frames():
#         if stop_event.is_set():
#             break
#     cv2.destroyAllWindows()
#     stop_event.set()  # Signal the video processing to stop
#     return render_template('index.html')


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/index')
# def index():
#     return render_template('index.html')


# @app.route("/video_feed")
# def video_feed():
#     global video_thread
#     if video_thread is None or not video_thread.is_alive():
#         video_thread = threading.Thread(target=video_processing)
#         video_thread.start()
#     return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# if __name__ == "__main__":
#     app.run(debug=True)

       
