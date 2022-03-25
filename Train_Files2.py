import os
import sys
import

os.chdir("/home/pi/opencv-face-recognition/")
sys.path.insert(1, '/home/pi/opencv-face-recognition/')
import train_model
command = "python train_model.py --embeddings /home/pi/opencv-face-recognition/output/embeddings.pickle --recognizer /home/pi/opencv-face-recognition/output/recognizer.pickle --le /home/pi/opencv-face-recognition/output/le.pickle"
#os.system("python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle")
os.system(command)