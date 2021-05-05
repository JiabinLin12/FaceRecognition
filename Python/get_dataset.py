import cv2
import os,sys,getopt
import time
import shutil
WIDTH = 640
HEIGHT = 480
WIDTH_ID = 3
HEIGHT_ID = 4
CAM_INDEX = 0
DATASET_NUM = 10
ESC = 27
PATH = 'dataset'
def dataset(argv):
	#create dataset folder
	if not os.path.exists(PATH):
		#shutil.rmtree(PATH,ignore_errors = False)
		os.makedirs(PATH)
	usr_face_id = int(argv)
	#Import face cascade
	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	#Initialize campera
	cap = cv2.VideoCapture(CAM_INDEX)
	
	#Video resolution
	cap.set(WIDTH_ID, WIDTH) 	
	cap.set(HEIGHT_ID, HEIGHT)
	
	#usr input id	
	data_count = 0;
	while True:
		#capture one frame
		ret, frame = cap.read()
		
		#convert to gray scale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#face detection, data set will only capture faces
		faces = faceCascade.detectMultiScale(
				gray, 
				scaleFactor=1.3, 
				minNeighbors = 5,
				minSize = (30,30))
		
		#draw box on face
		for(x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
			data_count+=1
			#save only faces to file
			cv2.imwrite("dataset/User." + str(usr_face_id) + '.' + str(data_count) + ".jpg", gray[y:y+h,x:x+w])
			#show image
			cv2.imshow('face', frame)
		
		#ESC to quit
		if((cv2.waitKey(30)&0xFF) == ESC):
			break;
		#30 Images captured
		elif data_count == DATASET_NUM:
			break;
	print("\n [INFO] Done captured user{0}".format(usr_face_id))
	time.sleep(2)
	data_count =0;
	cap.release()
	cv2.destroyAllWindows()
	print('\n [INFO] Done captured')
	

	
	
	
		
if __name__ == "__main__":
	dataset(sys.argv[1])
	
