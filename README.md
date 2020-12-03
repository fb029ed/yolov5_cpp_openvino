# yolov5_cpp_openvino
# 用c++实现了yolov5使用openvino的部署
> 在2020极市开发者榜单中取得老鼠识别赛题第四名
仅仅入门20天这个成绩我还是很满意的
现在被导师逼着转语音,以后可能用不到了
希望能给需要的朋友一些参考,节省一些踩坑的时间
选择导师很重要,祝各位能够碰到好的导师吧

# 部署流程
## 模型训练
### 1.首先获取yolov5工程

yolov5:
https://github.com/ultralytics/yolov5.git
```shell
git clone https://github.com/ultralytics/yolov5.git
```
写readme的时间是2020年12月3日,官方最新的releases是v3.1
在v3.0的版本中,官网有如下的声明
* August 13, 2020**: [v3.0 release](https://github.com/ultralytics/yolov5/releases/tag/v3.0): nn.Hardswish() activations, data autodownload, native AMP.

这里的Hardwish函数就是模型转换的最大障碍在openvino部署yolov5需要两次模型转换,v2.0是能够通过两次模型转换的最新版本,所以需要先将版本切换到v2.0的版本
```shell
git checkout v2.0 
```

### 2.训练准备

在yolov5的文件夹下/yolov5/models/目录下可以找到以下文件
> yolov5s.yaml
yolov5m.yaml
yolov5l.yaml

等用于描述模型结构的文件,在其中需要更改
* 类别信息
nc:需要识别的类别数量
* 锚框
anchors:通过kmeans等算法根据自己的数据集得出合适的锚框
这里需要注意:yolov5内部实现了锚框的自动计算训练过程默认使用自适应锚框计算.在4.执行训练中将提到禁止自动锚框计算的方法


### 3.执行训练
注意数据标注格式,不匹配的要转换,这里不赘述
```python
python /project/train/src_repo/yolov5/train.py --batch 16 --epochs 10 --data /project/train/src_repo/rat.yaml --cfg /project/train/src_repo/yolov5/models/yolov5s.yaml --weights ""
```
* 这里的rat.yaml需要根据数据自己实现,可以参考仓库的对应文件
* --weights 后面跟预训练模型的路径,如果是""则重新训练一个模型,这里需要注意,该预训练模型只支持v2.0版本的模型,如果现有的模型不支持可以用""从头训练,官网的v2.0版本模型实际上下载下载下来还是3.0版本的模型,真正的v2.0版本的模型理论上只有之前2.0就玩的老哥有,要的人不多我就不传了,偷个懒
* 禁止自适应anchor计算的方法
增加参数--noautoanchor
到训练完成之后就能获得pt格式的模型文件了

## 模型转换
### pt文件转onnx文件
### 1.模型转换程序版本设置

在/yolov5/models/export.py中
```python
torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=['classes', 'boxes'] if y is None else ['output'])
```
opset_version=12,将导致后面的模型装换有未实现的内容
因此设置为opset_version=10
该程序的作用是将pytorch训练输出的.pt格式的模型文件装换成.onnx格式的模型文件
tips:强烈推荐netron
https://github.com/lutzroeder/netron.git
该工具可以可视化onnx模型文件的结构,对于了解陌生网络的结构十分有帮助
### 2.转换过程
```python3
python3 ./yolov5/models/export.py --weights "+pt文件的路径+" --img 640 --batch 1"
```
执行这步之后可以获得onnx文件
### onnx模型到openvino格式的xml bin模型转换
### 1.openvino安装
略,主要需要确保是最新版本的openvino,旧版本的不一定支持一些算子,经测试2020r1可用
### 2.模型转换
可以参考
### [如何将模型转换成OpenVINO格式](https://docs.cvmart.net/#/guide?id=%e5%a6%82%e4%bd%95%e5%b0%86%e6%a8%a1%e5%9e%8b%e8%bd%ac%e6%8d%a2%e6%88%90openvino%e6%a0%bc%e5%bc%8f)
也可以参考本仓库openvino_cpp_code/cov.txt的内容
本步骤完成后可以获得xml文件和bin文件
之后使用openvino的c++接口调用即可
## 使用openvino进行部署
可以看程序openvino_cpp_code/my_detector.cpp内容
需要注意
该模型的输出还需要进行一些计算才能转化为常用的框信息,具体参考程序
其中的nms程序没有自己实现,调用的oopencv的实现,需要配置opencv环境,如果不想配置需要自行实现nms

## 其他
yolov5的仓库还是很活跃的遇到问题可以多去那里找找

