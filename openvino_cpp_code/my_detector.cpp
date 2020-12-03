#include "my_detector.h"
#define IMG_LEN 640
#define IMG_LEN_F  640.0f
static omp_lock_t lock_use;
static omp_lock_t s_lock;
MyDetector::MyDetector(){}

MyDetector::~MyDetector(){}

double MyDetector::sigmoid(double x){
    return (1 / (1 + exp(-x)));
}

/*
anchors:
  - [10,15, 13,24, 18,19]  # P3/8
  - [20,44, 29,26, 42,43]  # P4/16
  - [64,115, 139,111, 309,379]  # P5/32
*/
//TODO:此处数值需要根据anchor计算结果动态调整
vector<int> MyDetector::get_anchors(int net_grid){
    vector<int> anchors(6);
    int a80[6] = {10,15, 13,24, 18,19};
    int a40[6] = {20,44, 29,26, 42,43};
    int a20[6] = {64,115, 139,111, 309,379}; 
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


//注意此处的阈值是框和物体prob乘积的阈值
bool MyDetector::parse_yolov5(const Blob::Ptr &blob,int net_grid,float cof_threshold,
    vector<Rect>& o_rect,vector<float>& o_rect_cof){
    vector<int> anchors = get_anchors(net_grid);
    /*
    const int out_blob_h = static_cast<int>(blob->getTensorDesc().getDims()[2]);
    const int out_blob_w = static_cast<int>(blob->getTensorDesc().getDims()[3]);
    const int t1 = static_cast<int>(blob->getTensorDesc().getDims()[0]);
    const int t2 = static_cast<int>(blob->getTensorDesc().getDims()[1]);
 const int t3 = static_cast<int>(blob->getTensorDesc().getDims()[4]);


    cout << out_blob_h <<  "," << out_blob_w<< ","<<t1 << ","<<t2<< ","<<t3 <<  endl;
    */
    //cout << out_blob_h <<  "," << out_blob_w<< ","<<t1 << ","<<t2<< endl;
   LockedMemory<const void> blobMapped = as<MemoryBlob>(blob)->rmap();
   const float *output_blob = blobMapped.as<float *>();
   //TODO:待确认,80个类是85,一个类是6
   
    /*
    omp_set_num_threads(3);
    #pragma omp parallel for
    */
    int item_size = 6;
    size_t gi =  net_grid*item_size;
    size_t ggi = net_grid*gi;
    size_t anchor_n = 3;
    omp_set_num_threads(8);
    #pragma omp parallel for
    for(int i=0;i<net_grid;++i)
        for(int n=0;n<anchor_n;++n)
            for(int j=0;j<net_grid;++j)
            {
                double box_prob = output_blob[n*ggi + i*gi + j *item_size+ 4];
                box_prob = sigmoid(box_prob);
                //框置信度不满足则整体置信度不满足
                if(box_prob < cof_threshold)
                    continue;
                double obj_prob = output_blob[n*ggi + i*gi + j *item_size+ 5];
                obj_prob = sigmoid(obj_prob);
                float cof = box_prob * obj_prob;                
                //对于边框置信度小于阈值的边框,不关心其他数值,不进行计算减少计算量
                if(cof < cof_threshold)
                    continue;
                //注意此处输出为中心点坐标,需要转化为角点坐标
                double x = output_blob[n*ggi + i*gi + j*item_size + 0];
                double y = output_blob[n*ggi + i*gi + j*item_size + 1];
                double w = output_blob[n*ggi + i*gi + j*item_size + 2];
                double h = output_blob[n*ggi + i*gi + j *item_size+ 3];
                //只有一类,直接取得obj得分结果
                
                x = (sigmoid(x)*2 - 0.5 + j)*IMG_LEN_F/net_grid;
                y = (sigmoid(y)*2 - 0.5 + i)*IMG_LEN_F/net_grid;
                w = pow(sigmoid(w)*2,2) * anchors[n*2];
                h = pow(sigmoid(h)*2,2) * anchors[n*2 + 1];

                double r_x = x - w/2;
                double r_y = y - h/2;
                Rect rect = Rect(round(r_x),round(r_y),round(w),round(h));
                omp_set_lock(&s_lock); //获得互斥器
                o_rect.push_back(rect);
                o_rect_cof.push_back(cof);
                omp_unset_lock(&s_lock); //释放互斥器
            }
    return true;
    /*
    if(o_rect.size() == 0) return false;
    else return true;*/
}

//初始化
bool MyDetector::init(){
    //TODO:配置xml文件路径,路径有效性检查
    string xml = "/usr/local/ev_sdk/model/openvino/best.xml";
    //string xml = "/project/train/src_repo/firebest.xml";
    Core ie;
    auto cnnNetwork = ie.ReadNetwork(xml); 
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
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    omp_init_lock(&lock_use); // 初始化互斥锁
    omp_init_lock(&s_lock); // 初始化互斥锁
    return true;
}

//释放资源
bool MyDetector::uninit(){
    omp_destroy_lock(&lock_use); //销毁互斥器
    omp_destroy_lock(&s_lock); //销毁互斥器
    return true;
}

//从detect得到的xywh转换回到原图xywh
Rect MyDetector::detet2origin(const Rect& dete_rect,float rate_to,int top,int left){
    //detect坐标转换到内部纯图坐标
    int inside_x = dete_rect.x - left;
    int inside_y = dete_rect.y - top;
    int ox = round(float(inside_x)/rate_to);
    int oy = round(float(inside_y)/rate_to);
    int ow = round(float(dete_rect.width)/rate_to);
    int oh =  round(float(dete_rect.height)/rate_to);
    Rect origin_rect(ox,oy,ow,oh);
    return origin_rect;    
}

//处理图像获取结果
bool MyDetector::process_frame(Mat& inframe,vector<Object>& detected_objects){
    if(inframe.empty()){
        cout << "无效图片输入" << endl;
        return false;
    }
    //inframe在外部还需要用于生成输出图,不可直接使用
    
    //TODO:除了resize以外还需要检测是否需要进行rgb通道转换
    //resize(inframe,resize_img,Size(640,640));
    //cvtColor(resize_img,resize_img,COLOR_BGR2RGB);
    
    //以下为带边框图像生成
    int in_w = inframe.cols;
    int in_h = inframe.rows;
    int tar_w = IMG_LEN;
    int tar_h = IMG_LEN;
    //哪个缩放比例小选用哪个
    float r = min(float(tar_h)/in_h,float(tar_w)/in_w);
    int inside_w = round(in_w*r);
    int inside_h = round(in_h*r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    //内层图像resize
    Mat resize_img;
    resize(inframe,resize_img,Size(inside_w,inside_h));
    cvtColor(resize_img,resize_img,COLOR_BGR2RGB);
    
    padd_w = padd_w /2;
    padd_h = padd_h /2;
    //外层边框填充灰色
    int top = int(round(padd_h - 0.1)); 
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right =  int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top,bottom, left, right, BORDER_CONSTANT, Scalar(114,114,114));


    size_t img_size = IMG_LEN*IMG_LEN;
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    //nchw
    omp_set_num_threads(4);
    #pragma omp parallel for
    for(size_t row =0;row<IMG_LEN;row++){
        for(size_t col=0;col<IMG_LEN;col++){
            for(size_t ch =0;ch<3;ch++){
                blob_data[img_size*ch + row*IMG_LEN + col] = float(resize_img.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }
    //执行预测
    infer_request->Infer();
    //获取各层结果
    /*
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    int s[3] = {80,40,20};
    int i=0;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
       parse_yolov5(blob,s[i],0.5,origin_rect,origin_rect_cof);
        ++i;
    }
    */
    vector<Rect> origin_rect;
    vector<float> origin_rect_cof;
    int s[3] = {80,40,20};
    //大规模计算之前先收集指针
    vector<Blob::Ptr> blobs;
    for (auto &output : _outputinfo) {
        auto output_name = output.first;
        Blob::Ptr blob = infer_request->GetBlob(output_name);
        blobs.push_back(blob);
    }
    omp_set_num_threads(3);
    #pragma omp parallel for
    for(int i=0;i<blobs.size();++i){
        float th = 0.5;
        //小目标严格要求
        if(i == 0)
            th = 0.55;
        //大目标放宽要求
        else if(i==1)
            th = 0.45;
        else if(i==2)
            th = 0.40;
        //TODO:根据网格大小使用不同阈值
        vector<Rect> origin_rect_temp;
        vector<float> origin_rect_cof_temp;
        parse_yolov5(blobs[i],s[i],th,origin_rect_temp,origin_rect_cof_temp);
        //加入总的结果时加锁
        omp_set_lock(&lock_use); //获得互斥器
        origin_rect.insert(origin_rect.end(),origin_rect_temp.begin(),origin_rect_temp.end());
        origin_rect_cof.insert(origin_rect_cof.end(),origin_rect_cof_temp.begin(),origin_rect_cof_temp.end());
        omp_unset_lock(&lock_use); //释放互斥器
    }

    //后处理获得最终检测结果
    vector<int> final_id;
    //TODO:此处的阈值需要调整
    //TODO:第一参数0.5为当前最佳
    dnn::NMSBoxes(origin_rect,origin_rect_cof,0.40,0.5,final_id);
    //根据final_id获取最终结果
    //cout << final_id.size() << endl;
    for(int i=0;i<final_id.size();++i){
        Rect resize_rect= origin_rect[final_id[i]];
        Rect rawrect = detet2origin(resize_rect,r,top,left);
        detected_objects.push_back(Object{
            origin_rect_cof[final_id[i]],
            "rat",rawrect
        });
    }
    return true;
}

