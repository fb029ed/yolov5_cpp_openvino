#ifndef DETECTOR_H
#define DETECTOR_H
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Detector
{
public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;
    Detector();
    ~Detector();
    //初始化
    bool init(string xml_path,double cof_threshold,double nms_area_threshold);
    //释放资源
    bool uninit();
    //处理图像获取结果
    bool process_frame(Mat& inframe,vector<Object> &detected_objects);

private:
    double sigmoid(double x);
    vector<int> get_anchors(int net_grid);
    bool parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
        vector<Rect>& o_rect,vector<float>& o_rect_cof, vector<int> &classId);
    Rect detet2origin(const Rect& dete_rect,float rate_to,int top,int left);
    //存储初始化获得的可执行网络
    ExecutableNetwork _network;
    OutputsDataMap _outputinfo;
    string _input_name;
    //参数区
    string _xml_path;                             //OpenVINO模型xml文件路径
    double _cof_threshold;                //置信度阈值,计算方法是框置信度乘以物品种类置信度
    double _nms_area_threshold;  //nms最小重叠面积阈值
    string className[80] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		 "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		 "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		 "hair drier", "toothbrush" };
};
#endif
