import subprocess
import sys
import os


sys.path.insert(1, '/home/pi/opencv-face-recognition/')
import Recogvid
Recogvid_files = recog_files = subprocess.run(["Recogvid.py", "--detector", "face_detection_model", "--embedding-model", "openface_nn4.small2.v1.t7", "--recognizer", "output", "recognizer.pickle", "--le", "output/le.pickle"])
Recogvid_files.returncode