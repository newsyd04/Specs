from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import board
import digitalio
from PIL import Image, ImageDraw, ImageFont
import adafruit_ssd1306



oldName = " "





# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)


	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

#		if confidence < args["confidence"]:
#			# Clear display.
#			# Define the Reset Pin
#			oled_reset = digitalio.DigitalInOut(board.D4)
#			# Change these
#			# to the right size for your display!
#			WIDTH = 128
#			HEIGHT = 64  # Change to 64 if needed
#			BORDER = 5
#			# Use for I2C.
#			i2c = board.I2C()
#			oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C, reset=oled_reset)
#			oled.fill(0)
#			oled.show()
#			continue

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantificationdD4d44d4 of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]
			
			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 150)
			x = startX - 10 if startX - 10 > 10 else startX + 10
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(frame, text, (x, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			# Define the Reset Pin
			oled_reset = digitalio.DigitalInOut(board.D4)

			# Change these
			# to the right size for your display!
			WIDTH = 128
			HEIGHT = 32  # Change to 64 if needed
			BORDER = 5
			fontsize = 1

			# Use for I2C.
			i2c = board.I2C()
			oled = adafruit_ssd1306.SSD1306_I2C(WIDTH, HEIGHT, i2c, addr=0x3C, reset=oled_reset)

#			newName = name
#			if not newName == oldName:

			# Clear display.
			print("[INFO] Clearing OLED...")
			oled.fill(0)
			oled.show()

			# Create blank image for drawing.
			# Make sure to create image with mode '1' for 1-bit color.
			print("[INFO] Creating blank image on OLED...")
			image = Image.new("1", (oled.width, oled.height))

			# Get drawing object to draw on image.
			print("[INFO] Get drawing object...")
			draw = ImageDraw.Draw(image)
			
			# Load default font.
			#font = ImageFont.load_default()
			font = ImageFont.truetype("Quicksand_Light.otf", fontsize)
						
			fontsize = 1  # starting font size

			# portion of image width you want text width to be
			img_fraction = 0.50
				
			
			# Output name to OLED display
			print("[INFO] Fetch name for OLED...")			
			string2output = str(name)
			(font_width, font_height) = font.getsize(string2output)

			while font.getsize(string2output)[0] < img_fraction*image.size[0]:
				# iterate until the text size is just larger than the criteria
				fontsize += 1
				font = ImageFont.truetype("Quicksand_Light.otf", fontsize)
				
			#draw.text((10, 25), txt, font=font) # put the text on the image

			draw.text(
				(oled.width // 4 - font_width // 2, oled.height // 5 - font_height // 2),
				string2output,
				font=font,
				fill=255,
				)
			print("[INFO] Name for OLED is...",string2output)			
			# Display image
			print("[INFO] Display text on OLED...")
			oled.image(image)
			oled.show()
			#oldName = string2output
			
		#else:
			#oled.fill(0)
			#oled.show()
		

	# show the output frame
	cv2.imshow("Frame", frame)
		# if the `q` key was pressed, break from the loop

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		oled.fill(0)
		oled.show()
		break

			

