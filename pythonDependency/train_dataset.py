import cv2
import os
import numpy as np
from PIL import Image

TRAIN_PATH = 'train'
TRAIN_FILE = 'train/trainer.xml'
DATA_PATH = 'dataset'

def data_training():
	#import front face detection
	detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
	
	#get every frames from the dataset folder save all paths in imagePath list
	dataPaths = [os.path.join(DATA_PATH,f) for f in os.listdir(DATA_PATH)];
	#define face samples and ids container
	samples = []
	ids = []
	for i in dataPaths:
		#convert each image to grade scale
		PIL_img = Image.open(i).convert('L')
		img_numpy = np.array(PIL_img,'uint8')
		
		#get usr id from the path name
		Img_name = os.path.split(i)[1]
		ID = int(Img_name.split(".")[1])
		
		#append
		samples.append(img_numpy)
		ids.append(ID)
	return ids,samples


if __name__ == "__main__":
	
	#Use LBPH method for face recognizer
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	
	#create train folder
	if not os.path.exists(TRAIN_PATH):
		os.makedirs(TRAIN_PATH)
	
	print("\n [INFO] Data is training...")
	labels,faces = data_training()
	recognizer.train(faces,np.array(labels))
	
	#write to train path
	recognizer.write(TRAIN_FILE)
	
	#Exiting
	print("\n [INFO] {0} face trained".format(len(np.unique(labels))))
	
