import subprocess
import sys
import os
import shlex 


os.chdir("/home/pi/opencv-face-recognition/")
sys.path.insert(1, '/home/pi/opencv-face-recognition/')
print("test1")
import train_model
print("test2")
command = "python train_model.py --embeddings /home/pi/opencv-face-recognition/output/embeddings.pickle --recognizer /home/pi/opencv-face-recognition/output/recognizer.pickle --le /home/pi/opencv-face-recognition/output/le.pickle"
train_files = subprocess.Popen(bachCommand.split(), stdout=subprocess.PIPE)
output, error= process.communicate()
#print(os.getcwd())
#print(os.listdir())
#train_files = subprocess.call(["python", "train_model.py", "--embeddings", "/home/pi/opencv-face-recognition/output/embeddings.pickle", "--recognizer", "/home/pi/opencv-face-recognition/output/recognizer.pickle", "--le", "/home/pi/opencv-face-recognition/output/le.pickle"])
print("test3")
#train_files.returncode
print("test4")