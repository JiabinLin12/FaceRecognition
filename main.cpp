#include "FaceRecognition.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <unistd.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <semaphore.h>

#include <syslog.h>
#include <sys/time.h>
#include <sys/sysinfo.h>
#include <string>

#include <errno.h>
#include <iostream>
#include <time.h>
#include "logging.h"

#define NSEC_PER_SEC (1000000000)
#define USEC_PER_MSEC (1000)
#define NANOSEC_PER_SEC (1000000000)
#define NSEC_PER_MICROSEC (1000)
#define NSEC_PER_MSEC (1000000)
#define NUM_CPU_CORES (1)
#define TRUE (1)
#define FALSE (0)

#define NUM_THREADS (2 + 1)

int abortTest = FALSE;
int abortS1 = FALSE, abortS2 = FALSE;
string user_name;
string acc;
Point Pt1, Pt2, Pt3;
sem_t semS1, semS2;
double wcet1, wcet2, first_start;
struct timeval start_time_val;
struct timespec time_1 = {0, 0};
struct timespec time_2 = {0, 0};
void *Sequencer(void *threadp);
void *Service_1(void *threadp);
void *Service_2(void *threadp);
double getTimeMsec(void);
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t);
typedef struct
{
  VideoCapture cap;
  Ptr<LBPHFaceRecognizer> model;
  Mat frame;
} threadParams_t;


int main()
{
  int opt;
  vector<Mat> images;
  vector<int> labels;
  openlog("FaceRecognition", LOG_PID | LOG_NDELAY, LOG_USER);
  int i, rc, scope;
  cpu_set_t threadcpu;
  pthread_t threads[NUM_THREADS];
  threadParams_t threadParams;
  pthread_attr_t rt_sched_attr[NUM_THREADS];
  int rt_max_prio, rt_min_prio;
  struct sched_param rt_param[NUM_THREADS];
  struct sched_param main_param;
  pthread_attr_t main_attr;
  pid_t mainpid;
  cpu_set_t allcpuset;
  gettimeofday(&start_time_val, (struct timezone *)0);

  LOG(LOG_INFO, "System has %d processors configured and %d available.\n", get_nprocs_conf(), get_nprocs());

  CPU_ZERO(&allcpuset);

  for (i = 0; i < NUM_CPU_CORES; i++)
    CPU_SET(i, &allcpuset);

  LOG(LOG_INFO, "Using CPUS=%d from total available.\n", CPU_COUNT(&allcpuset));

  // initialize the sequencer semaphores
  if (sem_init(&semS1, 0, 0))
  {
    printf("Failed to initialize S1 semaphore\n");
    exit(-1);
  }
  if (sem_init(&semS2, 0, 0))
  {
    printf("Failed to initialize S2 semaphore\n");
    exit(-1);
  }

  mainpid = getpid();

  rt_max_prio = sched_get_priority_max(SCHED_FIFO);
  rt_min_prio = sched_get_priority_min(SCHED_FIFO);

  pthread_attr_getscope(&main_attr, &scope);

  if (scope == PTHREAD_SCOPE_SYSTEM)
  {
    LOG(LOG_INFO, "PTHREAD SCOPE SYSTEM\n");
  }
  else if (scope == PTHREAD_SCOPE_PROCESS)
  {
    LOG(LOG_INFO, "PTHREAD SCOPE PROCESS\n");
  }
  else
    LOG(LOG_INFO, "PTHREAD SCOPE UNKNOWN\n");

  LOG(LOG_INFO, "rt_max_prio=%d\n", rt_max_prio);
  LOG(LOG_INFO, "rt_min_prio=%d\n", rt_min_prio);

  for (i = 0; i < NUM_THREADS; i++)
  {

    CPU_ZERO(&threadcpu);
    CPU_SET(3, &threadcpu);

    rc = pthread_attr_init(&rt_sched_attr[i]);
    rc = pthread_attr_setinheritsched(&rt_sched_attr[i], PTHREAD_EXPLICIT_SCHED);
    rc = pthread_attr_setschedpolicy(&rt_sched_attr[i], SCHED_FIFO);
    rc = pthread_attr_setaffinity_np(&rt_sched_attr[i], sizeof(cpu_set_t), &threadcpu);

    rt_param[i].sched_priority = rt_max_prio - i;
    pthread_attr_setschedparam(&rt_sched_attr[i], &rt_param[i]);
  }

  LOG(LOG_INFO, "Service threads will run on %d CPU cores\n", CPU_COUNT(&threadcpu));

  threadParams.model = LBPHFaceRecognizer::create();

  //Idle stage
  while (1)
  {
    usage();
    cin >> opt;
    switch (opt)
    {
    case 1:
      addface(threadParams.model, images, labels);
      break;
    case 2:
      Face_Recognition(threadParams.model);
      break;
    case 3:
      ThreadCreate(threads, rt_sched_attr, threadParams);
      break;
    case 4:
      clear_data();
      break;
    default:
      cout << "Invalid input " << endl;
      return -1;
    }
  }
  closelog();
  return 0;
}



/*Create threads for RM FaceRecognition*/
void ThreadCreate(pthread_t threads[NUM_THREADS],
                  pthread_attr_t rt_sched_attr[NUM_THREADS],
                  threadParams_t threadParams)
{

  assert(pthread_create(&threads[1], &rt_sched_attr[1], Service_1, (void *)&(threadParams)) == 0);
  assert(pthread_create(&threads[2], &rt_sched_attr[2], Service_2, (void *)&(threadParams)) == 0);
  assert(pthread_create(&threads[0], &rt_sched_attr[0], Sequencer, (void *)&(threadParams)) == 0);

  for (int i = 0; i < NUM_THREADS; i++)
    pthread_join(threads[i], NULL);

  LOG(LOG_INFO, "Thread idx=1 ,Worst Case Execution Time %lf msec\n\n", wcet1);
  LOG(LOG_INFO, "Thread idx=2 ,Worst Case Execution Time %lf msec\n\n", wcet2);
}


void *Sequencer(void *threadp)
{
  struct timespec delay_time = {0, 16666666}; //16.66 msec, 60 Hz
  struct timespec remaining_time;
  double residual;
  int rc, delay_cnt = 0;
  unsigned long long seqCnt = 0;
  threadParams_t *threadParams = (threadParams_t *)threadp;
  threadParams->cap.open(0, CAP_ANY);
  if (!threadParams->cap.isOpened())
  {
    cout << "fail to open camera" << endl;
    return NULL;
  }
  threadParams->cap.set(CAP_PROP_FPS, 30);
  threadParams->cap.set(CAP_PROP_FRAME_WIDTH, 640);
  threadParams->cap.set(CAP_PROP_FRAME_HEIGHT, 480);
  for (int i = 0; i < 200; i++)
  {
    threadParams->cap >> (threadParams->frame);
    if (!(threadParams->frame).empty())
    {
      imshow(WINDOW_NAME, threadParams->frame);
      waitKey(1);
    }
  }

  first_start = getTimeMsec();

  do
  {
    delay_cnt = 0;
    residual = 0.0;
    do
    {
      rc = nanosleep(&delay_time, &remaining_time);

      if (rc == EINTR)
      {
        residual = remaining_time.tv_sec + ((double)remaining_time.tv_nsec / (double)NANOSEC_PER_SEC);

        if (residual > 0.0)
          LOG(LOG_ERR, "residual=%lf, sec=%d, nsec=%d\n", residual, (int)remaining_time.tv_sec, (int)remaining_time.tv_nsec);

        delay_cnt++;
      }
      else if (rc < 0)
      {
        perror("Sequencer nanosleep");
        exit(-1);
      }

    } while ((residual > 0.0) && (delay_cnt < 100));

    seqCnt++;
    //Servcie_1 = RT_MAX-1	@ 20 Hz
    if ((seqCnt % 3) == 0)
      sem_post(&semS1);

    // Service_2 = RT_MAX-2	@ 1 Hz 60
    if ((seqCnt % 60) == 0)
      sem_post(&semS2);

  } while (!abortTest);
  cout << "exiting sequencer" << endl;
  sem_post(&semS1);
  sem_post(&semS2);
  abortS1 = TRUE;
  abortS2 = TRUE;
  pthread_exit((void *)0);
}




void *Service_1(void *threadp)
{
  threadParams_t *threadParams = (threadParams_t *)threadp;
  struct timespec start_time = {0, 0}, finish_time = {0, 0}, thread_dt = {0, 0}, temp = {0, 0};
  int counter = 0;
  double start, end, et;
  struct timeval current_time_val, start_time_val;
  while (!abortS1)
  {
    sem_wait(&semS1);
    counter++;
    start = getTimeMsec();
    threadParams->cap >> (threadParams->frame);
    if (!user_name.empty())
    {
      putText(threadParams->frame, user_name, Point(30, 20), FONT, 1, Scalar(0, 255, 0), 2);
      //draw box in face area
      rectangle(threadParams->frame, Pt1, Pt2, Scalar(0, 255, 0), 2);
    }
    if (!acc.empty())
    {
      putText(threadParams->frame, acc, Point(30, 50), FONT, 1, Scalar(255, 0, 0), 2);
    }

    if (!(threadParams->frame).empty())
      imshow(WINDOW_NAME, threadParams->frame);
    if (waitKey(1) == 27)
      break;
    end = getTimeMsec();
    et = end - start;
    if (et > wcet1)
      wcet1 = et;
    //syslog(LOG_CRIT, "S1 %d end time %.3lf ms\n",counter,end);
    syslog(LOG_CRIT, "S1 %d start time %lf ms Execution time %lf ms\n", counter, start - first_start, et);
  }
  cout << "exiting S1" << endl;
  abortTest = TRUE;
  destroyWindow(WINDOW_NAME);
  pthread_exit((void *)0);
}

void *Service_2(void *threadp)
{
  threadParams_t *threadParams = (threadParams_t *)threadp;
  struct timespec start_time = {0, 0}, finish_time = {0, 0}, thread_dt = {0, 0}, temp = {0, 0};

  Mat face_roi, face_resized;
  Mat grayscale;
  double confidence = 0.0, scale = 3.0;
  double start, end, et;
  int label = -1, accuracy = 0;
  int counter = 0;
  struct timeval current_time_val, start_time_val;
  vector<Rect> faces;
  string default_name = "usr";
  //read training data
  threadParams->model->read(TRAINING_DATA_FILE);

  CascadeClassifier faceCascade;
  //read front face data
  faceCascade.load(FRONT_FACE_XML);

  while (!abortS2)
  {
    sem_wait(&semS2);
    start = getTimeMsec();
    counter++;
    threadParams->cap.read(threadParams->frame);
    //Convert frame from RBG to Gray
    cvtColor(threadParams->frame, grayscale, COLOR_BGR2GRAY);

    //Resize by a scale of 1/scale
    resize(grayscale, grayscale,
           Size(grayscale.size().width / scale,
                grayscale.size().height / scale));

    //Face detection
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 3, 0, Size(30, 30));

    //For each face detected
    for (Rect face : faces)
    {

      //get face region of insterest(face)
      face_roi = grayscale(face);

      //prediction,get label and confidence
      threadParams->model->predict(face_roi, label, confidence);

      //Bad accuracy, ignore
      if (confidence > 100)
      {
        user_name = "";
        acc = "";
        continue;
      }

      accuracy = 100 - (int)confidence;
      //Rect coordinate
      Pt1 = Point(cvRound(face.x * scale), cvRound(face.y * scale));
      Pt2 = Point(cvRound((face.x + face.width - 1) * scale), cvRound((face.y + face.height - 1) * scale));
      Pt3 = Point(cvRound(cvRound(face.x * scale)), cvRound((face.y + face.height - 1) * scale));

      user_name = default_name + to_string(label);
      acc = "P:" + to_string(accuracy);
    }
    gettimeofday(&current_time_val, (struct timezone *)0);
    end = getTimeMsec();
    et = end - start;
    if (et > wcet2)
      wcet2 = et;
    syslog(LOG_CRIT, "S2 %d start time %lf ms Execution time %lf ms\n", counter, start - first_start, et);
  }
  cout << "exiting S2" << endl;
  abortTest = TRUE;
  pthread_exit((void *)0);
}

int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec = stop->tv_sec - start->tv_sec;
  int dt_nsec = stop->tv_nsec - start->tv_nsec;

  if (dt_sec >= 0)
  {
    if (dt_nsec >= 0)
    {
      delta_t->tv_sec = dt_sec;
      delta_t->tv_nsec = dt_nsec;
    }
    else
    {
      delta_t->tv_sec = dt_sec - 1;
      delta_t->tv_nsec = NSEC_PER_SEC + dt_nsec;
    }
  }
  else
  {
    if (dt_nsec >= 0)
    {
      delta_t->tv_sec = dt_sec;
      delta_t->tv_nsec = dt_nsec;
    }
    else
    {
      delta_t->tv_sec = dt_sec - 1;
      delta_t->tv_nsec = NSEC_PER_SEC + dt_nsec;
    }
  }

  return (1);
}
double getTimeMsec(void)
{
  struct timespec event_ts = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &event_ts);
  return ((event_ts.tv_sec) * 1000.0) + ((event_ts.tv_nsec) / 1000000.0);
}