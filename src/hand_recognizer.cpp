#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/Point.h>
#include <cv_nxtdrive/HandRect.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include <math.h>

namespace enc = sensor_msgs::image_encodings;
using namespace cv;
using namespace std;
const std::string cascade_name = "hand_recognition/palm3.xml";
ros::Publisher recog_pub;

CascadeClassifier cascade(cascade_name.c_str());

int prepare_env()
{
    const char *path = getenv("ROS_HOME");
    if (path != NULL)
        return chdir(path);
    return 0;
}

//#define PROFILE_RECOGNIZE
#ifdef PROFILE_RECOGNIZE
void deltat_stat(struct timespec *deltat, int deltat_size, int& mean_us, int& std_us)
{
    long total_us = 0;
    for (int i = 0; i < deltat_size; i++)
        total_us += deltat[i].tv_sec * 1000000 + deltat[i].tv_nsec / 1000;
    //cout << "total: " << total_ms;
    mean_us = total_us / deltat_size;
    total_us = 0;
    for (int i = 0; i < deltat_size; i++) {
        long d = deltat[i].tv_sec * 1000000 + deltat[i].tv_nsec / 1000 - mean_us;
        long d2 = d*d;
        total_us += d2;
    }
    //cout << "total: " << total_ms;
    std_us = sqrt(total_us / deltat_size);
}
#endif

int recognize(const Mat& img, cv_nxtdrive::HandRect& hr)
{
#ifdef PROFILE_RECOGNIZE
    struct timespec tv0;
    struct timespec tv_cur;
    const int DELTAT_MAX = 1000;
    static struct timespec deltat[DELTAT_MAX];
    static const int deltat_size = sizeof(deltat) / sizeof(struct timespec);
    static int deltat_i = 0;
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv0);
#endif
    vector<Rect> objects;
    cascade.detectMultiScale(img, objects);
    int i;
    int scale = 1;
    for (i = 0; i < (int)objects.size(); i++ ) {
        hr.p1.x = objects[i].x * scale;
        hr.p2.x = (objects[i].x + objects[i].width) * scale;
        hr.p1.y = objects[i].y * scale;
        hr.p2.y = (objects[i].y + objects[i].height) * scale;
    }

#ifdef PROFILE_RECOGNIZE
    clock_gettime(CLOCK_MONOTONIC_RAW, &tv_cur);
    if (deltat_i < deltat_size) {
        deltat[deltat_i].tv_sec = tv_cur.tv_sec - tv0.tv_sec;
        deltat[deltat_i].tv_nsec = tv_cur.tv_nsec - tv0.tv_nsec;
        //cout << deltat[deltat_i].tv_sec << " " << deltat[deltat_i].tv_nsec << endl;
    } else if (deltat_i == deltat_size) {
        int mean_us;
        int std_us;
        deltat_stat(deltat, deltat_size, mean_us, std_us);
        cout << "Mean: " << mean_us << " Std: " << std_us << endl;
    }
    deltat_i++;
#endif
    return objects.size();
}

/*
void recognize(Mat img, cv_nxtdrive::HandRect& hr)
{
    static CvMemStorage* storage = 0;
    static CvHaarClassifierCascade* cascade = 0;
    IplImage img_ipl = img;
    int i;
    int scale = 1;
    cascade = (CvHaarClassifierCascade *)cvLoad(cascade_name.c_str(), 0, 0, 0);
    if (! cascade) {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        return;
    }
    storage = cvCreateMemStorage(0);
    cvClearMemStorage(storage);
    CvSeq* hands = cvHaarDetectObjects(&img_ipl, cascade, storage,
                                       1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
                                       cvSize(40, 40));
    for (i = 0; i < (hands ? hands->total : 0); i++ ) {
        CvRect* r = (CvRect *)cvGetSeqElem(hands, i);
        hr.p1.x = r->x * scale;
        hr.p2.x = (r->x + r->width) * scale;
        hr.p1.y = r->y * scale;
        hr.p2.y = (r->y + r->height) * scale;
    }
}
*/

void image_cb(const sensor_msgs::ImageConstPtr& msg)
{
    const int RECOG_FRAME_WINDOW = 1;
    cv_bridge::CvImagePtr cv_ptr;
    static int frame_cnt = -1;
    int n_recog;
    geometry_msgs::Point no_pt;
    no_pt.x = -1.0;
    no_pt.y = -1.0;
    no_pt.z = -1.0;

    frame_cnt++;
    if (frame_cnt % RECOG_FRAME_WINDOW != 0)
        return;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv_nxtdrive::HandRect hr;
    n_recog = recognize(cv_ptr->image, hr);
    if (n_recog <= 0) {
        hr.p1 = no_pt;
        hr.p2 = no_pt;
    }
    recog_pub.publish(hr);
}

void run_node(int argc, char **argv)
{
    const std::string RGB_TOPIC = "/camera/rgb/image_color";
    const std::string RECOG_TOPIC = "/hand_recognizer/pos";
    ros::init(argc, argv, "hand_recognizer");
    ros::NodeHandle nh;
    recog_pub = nh.advertise<cv_nxtdrive::HandRect>(RECOG_TOPIC, 100);
    ros::Subscriber img_subs = nh.subscribe(RGB_TOPIC, 100, image_cb);
    ros::spin();
}

int main(int argc, char **argv)
{
    if (prepare_env() != 0) {
        ROS_ERROR("Cannot set current dir to $ROS_HOME");
        return -1;
    }
    run_node(argc, argv);
}   
