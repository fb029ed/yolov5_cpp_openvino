#include "detector.h"

Detector::Detector(){}

Detector::~Detector(){}

//注意此处的阈值是框和物体prob乘积的阈值
bool Detector::parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
    vector<Rect>& o_rect,vector<float>& o_rect_cof,
    vector<int> &classId){
    vector<int> anchors = get_anchors(net_grid);
   LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
   const float *output_blob = blobMapped.as<float *>();
   //80个类是85,一个类是6,n个类是n+5
   //int item_size = 6;
   int item_size = 85;
    size_t anchor_n = 3;
    for(int n=0;n<anchor_n;++n)
        for(int i=0;i<net_grid;++i)
            for(int j=0;j<net_grid;++j)
            {
                double box_prob = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 4];
                box_prob = sigmoid(box_prob);
                //框置信度不满足则整体置信度不满足
                if(box_prob < cof_threshold)
                    continue;
                
                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 0];
                double y = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 1];
                double w = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j*item_size + 2];
                double h = output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ 3];
               
                double max_prob = 0;
                int idx=0;
                // for(int t=5;t<85;++t){
                //     double tp= output_blob[n*net_grid*net_grid*item_size + i*net_grid*item_size + j *item_size+ t];
                //     tp = sigmoid(tp);
                //     if(tp > max_prob){
                //         max_prob = tp;
                //         idx = t;
                //     }
                // }

                for (int t = 0; t < item_size; ++t) {
					double tp = sigmoid(output_blob[n*net_grid*net_grid*item_size + i * net_grid*item_size + j * item_size + t]);
					if (tp > max_prob) {
						max_prob = tp;
						idx = t-5;
					}
				}
                float cof = box_prob * max_prob;                
                //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                if(cof < cof_threshold)
                    continue;

                x = (sigmoid(x)*2 - 0.5 + j)*640.0f/net_grid;
                y = (sigmoid(y)*2 - 0.5 + i)*640.0f/net_grid;
                w = pow(sigmoid(w)*2,2) * anchors[n*2];
                h = pow(sigmoid(h)*2,2) * anchors[n*2 + 1];

                double r_x = x - w/2;
                double r_y = y - h/2;
                Rect rect = Rect(round(r_x),round(r_y),round(w),round(h));
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                classId.push_back(idx);
            }
    if(o_rect.size() == 0) return false;
    else return true;
}

//初始化
bool Detector::init(string xml_path,double cof_threshold,double nms_area_threshold){
    _xml_path = xml_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(_xml_path); 
    //输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //输出设置
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}

//释放资源
bool Detector::uninit(){
    return true;
}

//处理图像获取结果
bool Detector::process_frame(Mat& inframe,vector<Object>& detected_objects){
    if(inframe.empty()){
        cout << "无效图片输入" << endl;
        return false;
    }
    resize(inframe,inframe,Size(640,640));
    cvtColor(inframe,inframe,COLOR_BGR2RGB);
    size_t img_size = 640*640;
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    //nchw
    for(size_t row =0;row<640;row++){
        for(size_t col=0;col<640;col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*640 + col] = float(inframe.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }
    //执行预测
    infer_request->Infer();
    //获取各层结果
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    vector<int> classId;
    int s[3] = {80,40,20};
    int i=0;
    for (auto &output : _outputinfo) {
        //用于保存解析结果的临时vector 
		vector<cv::Rect> origin_rect_temp;
		vector<float> origin_rect_cof_temp;
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        parse_yolov5(blob,s[i],_cof_threshold,origin_rect,origin_rect_cof, classId);
        origin_rect.insert(origin_rect.end(), origin_rect_temp.begin(), origin_rect_temp.end());
		origin_rect_cof.insert(origin_rect_cof.end(), origin_rect_cof_temp.begin(), origin_rect_cof_temp.end());

        ++i;
    }
    //后处理获得最终检测结果
    vector<int> final_id;
    dnn::NMSBoxes(origin_rect,origin_rect_cof,_cof_threshold,_nms_area_threshold,final_id);
    //根据final_id获取最终结果
    for(int i=0;i<final_id.size();++i){
        Rect resize_rect= origin_rect[final_id[i]];
        detected_objects.push_back(Object{
            origin_rect_cof[final_id[i]],
            className[classId[final_id[i]]],resize_rect
        });
        cout << className[classId[final_id[i]]] << endl;
    }
    return true;
}

//以下为工具函数
double Detector::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

vector<int> Detector::get_anchors(int net_grid){
    vector<int> anchors(6);
    int a80[6] = {10,13, 16,30, 33,23};
    int a40[6] = {30,61, 62,45, 59,119};
    int a20[6] = {116,90, 156,198, 373,326}; 
    if(net_grid == 80){
        anchors.insert(anchors.begin(),a80,a80 + 6);
    }
    else if(net_grid == 40){
        anchors.insert(anchors.begin(),a40,a40 + 6);
    }
    else if(net_grid == 20){
        anchors.insert(anchors.begin(),a20,a20 + 6);
    }
    return anchors;
}


