#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <filesystem>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/face.hpp>
#include "opencv2/highgui.hpp"

#include <fstream>
#include <sstream>

#include <sched.h>
#include <time.h>
#include <syslog.h>
#include <pthread.h>
#include <semaphore.h>

#define GET_DATASET_CMD 	"python3 get_dataset.py "
#define TRAIN_DATASET_CMD 	"python3 train_dataset.py"
#define CLEAR_DATA_CMD 		"./clean_dataset.sh"
#define TRAINING_DATA_PATH 	"train"
#define TRAINING_DATA_FILE 	"train/trainer.xml"
#define FRONT_FACE_XML 		"haarcascade_frontalface_default.xml"
#define PATH  				"dataset/"
#define IMG_PATH 			"dataset/*.jpg"
#define WINDOW_NAME 		"RT CAM"



#define HRES 640
#define VRES 480
#define CAM_DEV_IDX 0

#define FONT FONT_HERSHEY_SIMPLEX

using namespace cv;
using namespace std;
using namespace cv::face;
namespace fs = std::filesystem;

void Face_Recognition(Ptr<LBPHFaceRecognizer> &model);
void addface(Ptr<LBPHFaceRecognizer> &model,vector<Mat> &images, vector<int> &ids);
void clear_data();
void usage();
