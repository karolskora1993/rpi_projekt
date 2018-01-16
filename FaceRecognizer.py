activate_this = "/home/pi/.virtualenvs/cv/bin/activate_this.py"
execfile(activate_this, dict(__file__=activate_this))
import cv2
import cv2
import time
import json
from RequestOrganizer import RequestOrganizer
from ModelHandler import ModelHandler

model_handler = ModelHandler()


class FaceDetector:
    def __init__(self):
        self.configs = {}
        self.train_mode = False

        with open('config.json') as configs:
            self.configs = json.loads(configs.read())

        print("LOADING VIDEO CAMERA")
        self.OpenCVCapture = cv2.VideoCapture(0)

    def deviceCommandsCallback(self, topic, payload):
        print("Received command data: %s" % (payload))
        newSettings = json.loads(payload.decode("utf-8"))


face_detector = FaceDetector()
model = cv2.face.EigenFaceRecognizer_create(threshold=face_detector.configs["ClassifierSettings"]["predictionThreshold"])
model.read(face_detector.configs["ClassifierSettings"]["Model"])
print("LOADED STREAM & MODEL")
time_sleep = 3

while True:
    if face_detector.train_mode:
        print("TRAINING MODE")
        model_handler.process_training_data()
        model_handler.trainModel()
        model_handler.train_mode = False
    else:
        print("DETECTION MODE")
        try:

            ret, frame = face_detector.OpenCVCapture.read()
            print("photo taken")
            if not ret:
                print("not ret")
                continue

            currentImage, detected = model_handler.captureAndDetect(frame)
            if detected is None:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            x, y, w, h = detected
            crop = model_handler.resize(model_handler.crop(image, x, y, w, h))
            label, confidence = model.predict(crop)

            if float(confidence) < 1000:
                print("Person " + str(label) + " Confidence " + str(confidence))
                time_sleep = 3

            else:
                description = "Person not recognised"
                time_sleep = 20
                print(description)
                cv2.imwrite("image.png", image)
                file = open("image.png","rb")
                RequestOrganizer.sendRequest(face_detector.configs["ApiSettings"]["URL"])
                print("image sent")
                print("setting time sleep to 20 sec")

        except cv2.error as e:
            print(e)
        finally:
            time.sleep(time_sleep)



face_detector.OpenCVCapture.release()
cv2.destroyAllWindows()
