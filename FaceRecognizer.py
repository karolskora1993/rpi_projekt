import cv2
import sys
import os
import time
import json
from datetime import datetime

from ModelHandler import ModelHandler

model_handler = ModelHandler()

class FaceDetector():
	
	def __init__(self):
		
		self.configs = {}
		self.train_mode = False 
		
		with open('config.json') as configs:
			self.configs = json.loads(configs.read())
			
		print("LOADING VIDEO CAMERA")
		self.OpenCVCapture = cv2.VideoCapture(0)
		
	def deviceCommandsCallback(self,topic,payload):
		
		print("Received command data: %s" % (payload))
		newSettings = json.loads(payload.decode("utf-8"))
		
		
face_detector = FaceDetector()
model = cv2.face.createEigenFaceRecognizer(threshold=TASS.configs["ClassifierSettings"]["predictionThreshold"])
model.load(TASS.configs["ClassifierSettings"]["Model"])
print("LOADED STREAM & MODEL")
while 
	if face_detector.train:
		print("TRAINING MODE")
		model_handler.processTrainingData()
		model_handler.trainModel()
		.train=False		
	else:
			
		try:
			
			ret, frame = face_detector.OpenCVCapture.read()
			if not ret: continue
			
			currentImage,detected = model_handler.captureAndDetect(frame)
			if detected is None:
				continue
				
			image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			x, y, w, h = detected
			crop = model_handler.resize(model_handler.crop(image, x, y, w, h))
			label,confidence = model.predict(crop)

			if label:

				print("Person " + str(label) + " Confidence " +str(confidence))

			else:

				print("Person not recognised " + str(label) + " Confidence "+str(confidence));

			time.sleep(1)
		
		except cv2.error as e:
			print(e)

face_detector.OpenCVCapture.release()
cv2.destroyAllWindows()