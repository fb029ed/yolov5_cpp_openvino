FROM uhub.service.ucloud.cn/eagle_nest/cuda10.1-cudnn7.5.1-dev-ubuntu16.04-opencv4.1.1-torch1.4.0-openvino2020r2-workspace

# 创建默认目录，训练过程中生成的模型文件、日志、图必须保存在这些固定目录下，训练完成后这些文件将被保存
RUN mkdir -p /project/train/src_repo && mkdir -p /project/train/result-graphs && mkdir -p /project/train/log && mkdir -p /project/train/models

# 安装训练环境依赖端软件，请根据实际情况编写自己的代码
COPY ./ /project/train/src_repo/
RUN python3.6 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /project/train/src_repo/requirements.txt

