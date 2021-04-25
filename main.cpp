#include "FaceRecognition.h"



int main(){
	int opt;
	vector<Mat> images;
	vector<int> labels;
	Ptr<LBPHFaceRecognizer> model = LBPHFaceRecognizer::create();	
	//Idle stage
	while(1)
	{
		usage();
		cin>>opt;
		switch (opt)
		{
			case 1:
				addface(model,images,labels);
				break;
			case 2:
				Face_Recognition(model);
				break;
			case 3:
				clear_data();
				break;
			default:
				cout << "Invalid input " <<endl;
				return -1;
		}
	}
	return 0;
}
