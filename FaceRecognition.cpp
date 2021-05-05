#include "FaceRecognition.h" 


void train_dataset(Ptr<LBPHFaceRecognizer> &model,vector<Mat> &images, vector<int> &ids){
	struct dirent *entry;
	DIR *dp;
	vector<String> fn;
	string delimiter = ".",s;
	size_t pos = 0;
	int ret,c=0,id;
	//check dataset exist
	dp = opendir(PATH);
	if(dp == NULL){
		perror("\n [ERROR] Open dir failed");
		return;
	}
	
	//Read dataset
	glob(IMG_PATH,fn);
	for(auto f:fn){
		Mat img = imread(f, IMREAD_GRAYSCALE);
		images.push_back(img);
		s = f;
		while((pos = s.find(delimiter))!=string::npos){
			c++;
			//save id
			if(c==2){				
				id = stoi(s.substr(0,pos));
				ids.push_back(id);
				break;
			}
			s.erase(0, pos+delimiter.length());
		}
		c=0;
	}
	
	//train data
	model->train(images, ids);
	//create train folder if not exist
	ret = system("mkdir -p train");
	if(ret != 0 || !WIFEXITED(ret)){
		cout << "Subprocess to create dir aborted" << endl;
		return;
	}
	//write .xml to train folder
	model->save(TRAINING_DATA_FILE);
	cout <<"\n [INFO] Training finish" << endl;
	closedir(dp);
}


void Face_Recognition(Ptr<LBPHFaceRecognizer> &model){
	Point Pt1, Pt2,Pt3;
	Mat face_roi,face_resized;
	Mat frame,grayscale;
	double confidence = 0.0,scale = 3.0;
	int label = -1, accuracy = 0;
	vector<Rect> faces;	
	VideoCapture cap(0);
	string default_name = "usr";
	//read training data
	model->read(TRAINING_DATA_FILE);
	
	CascadeClassifier faceCascade; 
	//read front face data
	faceCascade.load(FRONT_FACE_XML);
	
	if(!cap.isOpened()){
		cout << "fail to open camera" <<endl;
		return ;
	}
	cap.set(CAP_PROP_FPS,30);
	cout << "\n [INFO] Starting Face recognition" << endl;
	while(1){
		//Capture one frame
		cap >> frame;
		
		//Convert frame from RBG to Gray
		cvtColor(frame,grayscale, COLOR_BGR2GRAY);
		
		//Resize by a scale of 1/scale		
		resize(grayscale,grayscale, 
				Size(grayscale.size().width / scale, 
				grayscale.size().height / scale));
		
		//Face detection
		faceCascade.detectMultiScale(grayscale,faces, 1.1, 3, 0, Size(30,30));
		
		//For each face detected
		for(Rect face : faces){
			
			
			//get face region of insterest(face)
		    face_roi = grayscale(face);
		    
		    //prediction,get label and confidence			
			model->predict(face_roi,label, confidence);
			
			//Bad accuracy, ignore
			if(confidence > 100)
				continue;
			
			accuracy = 100 - (int)confidence;
			//Rect coordinate
			Pt1 = Point(cvRound(face.x*scale),cvRound(face.y*scale));
		    Pt2 = Point(cvRound((face.x+face.width-1)*scale),cvRound((face.y+face.height-1)*scale));
			Pt3 = Point(cvRound(cvRound(face.x*scale)),cvRound((face.y+face.height-1)*scale));
			//draw box in face area
			rectangle(frame, Pt1, Pt2, Scalar(0,255,0), 2);
	  		
			putText(frame, default_name+to_string(label), Pt1, FONT, 1, Scalar(255,255,255),2);
			putText(frame, "P:"+to_string(accuracy)+"%",Pt3, FONT,1,Scalar(255,255,0),2);
	  		cout<<"confidence "<<confidence << " label" << label << endl;
			
		}
       
		imshow(WINDOW_NAME,frame);
		if(waitKey(10)==27)
			break;
	}
	//ReleaseCapture(&cap);
    destroyWindow(WINDOW_NAME);
	
}

void addface(Ptr<LBPHFaceRecognizer> &model,vector<Mat> &images, vector<int> &ids){
	int id, ret;
	cout << "User ID: ";
	scanf(" %d", &id);
	string tmp = GET_DATASET_CMD + to_string(id);
	const char *cmd = tmp.c_str();
	//call get_data.py
	ret = system(cmd);
	if(ret != 0 || !WIFEXITED(ret)){
		cout << "Subprocess aborted" << endl;
		return;
	}
	train_dataset(model,images,ids);
}

void clear_data(){
	//call clean_dataset.sh
	int ret;
	ret = system(CLEAR_DATA_CMD);
	if(ret != 0 || !WIFEXITED(ret)){
		cout << "Subprocess aborted" << endl;
		return;
	}
}


void usage(){
	cout<<"1. Added face"<<endl;
	cout<<"2. BF_FaceRecognition"<<endl;
	cout<<"3. RM_FaceRegnition"<<endl;
	cout<<"4. Clear dataset"<<endl;
}

