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
model = cv2.face.createEigenFaceRecognizer(threshold=face_detector.configs["ClassifierSettings"]["predictionThreshold"])
model.load(face_detector.configs["ClassifierSettings"]["Model"])
print("LOADED STREAM & MODEL")

while True:
    if face_detector.train:
        print("TRAINING MODE")
        model_handler.processTrainingData()
        model_handler.trainModel()
        model.train = False
    else:

        try:

            ret, frame = face_detector.OpenCVCapture.read()
            if not ret:
                continue

            currentImage, detected = model_handler.captureAndDetect(frame)
            if detected is None:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            x, y, w, h = detected
            crop = model_handler.resize(model_handler.crop(image, x, y, w, h))
            label, confidence = model.predict(crop)

            if label:

                print("Person " + str(label) + " Confidence " + str(confidence))

            else:
                description = "Person not recognised " + str(label) + " Confidence " + str(confidence)
                print(description)
                url = face_detector.configs["ApiSettings"]["URL"]
                send_data = {
                    "image": image,
                    "description": description
                }
                RequestOrganizer.sendRequest(face_detector.configs["ApiSettings"]["URL"], )

            time.sleep(1)

        except cv2.error as e:
            print(e)


face_detector.OpenCVCapture.release()
cv2.destroyAllWindows()
