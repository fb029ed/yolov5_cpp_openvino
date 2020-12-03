#ifndef MY_DETECTOR_H
#define MY_DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class MyDetector
{
public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;
    MyDetector();
    ~MyDetector();
    //初始化
    bool init();
    //释放资源
    bool uninit();
    //处理图像获取结果
    bool process_frame(Mat& inframe,vector<Object> &detected_objects);
private:
    double sigmoid(double x);
    vector<int> get_anchors(int net_grid);
    bool parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
        vector<Rect>& o_rect,vector<float>& o_rect_cof);
    Rect detet2origin(const Rect& dete_rect,float rate_to,int top,int left);

    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;
};
#endif
