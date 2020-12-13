/**
 * 示例代码：实现ji.h定义的图像接口，开发者需要根据自己的实际需求对接口进行实现
 */

#include <cstdlib>
#include <cstring>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "cJSON.h"
#include "ji.h"

#include "my_detector.h"

#ifndef EV_SDK_DEBUG
#define EV_SDK_DEBUG 1
#endif
char *jsonResult = nullptr; // 用于存储算法处理后输出到JI_EVENT的json字符串，根据ji.h的接口规范，接口实现需要负责释放该资源
cv::Mat outputFrame;

/**
 * 使用predictor对输入图像inFrame进行处理
 *
 * @param[in] predictor 算法句柄
 * @param[in] inFrame 输入图像
 * @param[in] args 处理当前输入图像所需要的输入参数，例如在目标检测中，通常需要输入ROI，由开发者自行定义和解析
 * @param[out] outFrame 输入图像，由内部填充结果，外部代码需要负责释放其内存空间
 * @param[out] event 以JI_EVENT封装的处理结果
 * @return 如果处理成功，返回JISDK_RET_SUCCEED
 */
int processMat(MyDetector *detector,  cv::Mat &inFrame, const char* args, cv::Mat &outFrame, JI_EVENT &event) {
    // 处理输入图像
    if (inFrame.empty()) {
        return JISDK_RET_FAILED;
    }

    // 针对每个ROI进行算法处理
    std::vector<MyDetector::Object> detectedObjects;
    std::vector<cv::Rect> detectedTargets;
    // 算法处理
    bool process_success = detector->process_frame(inFrame, detectedObjects);
    if (!process_success) {
        return JISDK_RET_FAILED;
    }

    // 创建输出图
    inFrame.copyTo(outFrame);

    // 将结果封装成json字符串
    cJSON *rootObj = cJSON_CreateObject();
    cJSON *pedestrianObj = cJSON_CreateArray();
        //Mat tempimg = inFrame.clone();
        /*
        resize(inFrame,tempimg,Size(640,640));
        */    
    for (auto &obj : detectedObjects) {
        cJSON *odbObj = cJSON_CreateObject();
        int xmin = obj.rect.x;
        int ymin = obj.rect.y;
        int xmax = xmin + obj.rect.width;
        int ymax = ymin + obj.rect.height;
        cJSON_AddItemToObject(odbObj, "xmin", cJSON_CreateNumber(xmin));
        cJSON_AddItemToObject(odbObj, "ymin", cJSON_CreateNumber(ymin));
        cJSON_AddItemToObject(odbObj, "xmax", cJSON_CreateNumber(xmax));
        cJSON_AddItemToObject(odbObj, "ymax", cJSON_CreateNumber(ymax));
        cJSON_AddItemToObject(odbObj, "confidence", cJSON_CreateNumber(obj.prob));
        cJSON_AddItemToObject(odbObj, "name", cJSON_CreateString(obj.name.c_str()));
        cJSON_AddItemToArray(pedestrianObj, odbObj);
        //test
        //rectangle(tempimg, obj.rect, Scalar(255, 0, 0),1, LINE_8,0);
    }
        //imwrite("tt.jpg",tempimg);
    
    cJSON_AddItemToObject(rootObj, "objects", pedestrianObj);

    char *jsonResultStr = cJSON_Print(rootObj);
    int jsonSize = strlen(jsonResultStr);
    if (jsonResult == nullptr) {
        jsonResult = new char[jsonSize + 1];
    } else if (strlen(jsonResult) < jsonSize) {
        free(jsonResult);   // 如果需要重新分配空间，需要释放资源
        jsonResult = new char[jsonSize + 1];
    }
    strcpy(jsonResult, jsonResultStr);

    event.json = jsonResult;
    event.code = JISDK_RET_SUCCEED;

    if (rootObj)
        cJSON_Delete(rootObj);
    if (jsonResultStr)
        free(jsonResultStr);

    return JISDK_RET_SUCCEED;
}

int ji_init(int argc, char **argv) {
    return JISDK_RET_SUCCEED;
}


void *ji_create_predictor(int pdtype) {
    auto * detector = new MyDetector();

    bool init_success = detector->init();
    if (!init_success) {
        return nullptr;
    }
    LOG(INFO) << "SamplePredictor init OK.";
    return detector;
}

void ji_destroy_predictor(void *predictor) {
    if (predictor == NULL) return;
    auto *detector = reinterpret_cast<MyDetector *>(predictor);
    detector->uninit();
    delete detector;

    if (jsonResult) {
        free(jsonResult);
        jsonResult = nullptr;
    }
}

int ji_calc_frame(void *predictor, const JI_CV_FRAME *inFrame, const char *args,
                  JI_CV_FRAME *outFrame, JI_EVENT *event) {
    if (predictor == NULL || inFrame == NULL) {
        return JISDK_RET_INVALIDPARAMS;
    }

    auto *detector = reinterpret_cast<MyDetector *>(predictor);
    cv::Mat inMat(inFrame->rows, inFrame->cols, inFrame->type, inFrame->data, inFrame->step);
    if (inMat.empty()) {
        return JISDK_RET_FAILED;
    }
    int processRet = processMat(detector, inMat, args, outputFrame, *event);

    if (processRet == JISDK_RET_SUCCEED) {
        if ((event->code != JISDK_CODE_FAILED) && (!outputFrame.empty()) && (outFrame)) {
            outFrame->rows = outputFrame.rows;
            outFrame->cols = outputFrame.cols;
            outFrame->type = outputFrame.type();
            outFrame->data = outputFrame.data;
            outFrame->step = outputFrame.step;

        }
    }
    return processRet;
}

int ji_calc_buffer(void *predictor, const void *buffer, int length, const char *args, const char *outFile,
                   JI_EVENT *event) {
    return JISDK_RET_UNUSED;
}

int ji_calc_file(void *predictor, const char *inFile, const char *args, const char *outFile, JI_EVENT *event) {
    return JISDK_RET_UNUSED;
}

int ji_calc_video_file(void *predictor, const char *infile, const char* args,
                       const char *outfile, const char *jsonfile) {
    return JISDK_RET_UNUSED;
}

void ji_reinit() {}