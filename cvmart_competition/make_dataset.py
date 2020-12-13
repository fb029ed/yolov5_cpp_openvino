import os
import random
output_base_path = "/project/train/src_repo/dataset/"
output_base_path_img = output_base_path + "images" 
output_base_path_label = output_base_path + "labels" 
output_base_path_img_t = output_base_path + "images" + "/train"
output_base_path_img_v = output_base_path + "images" + "/val"
output_base_path_label_t = output_base_path + "labels" + "/train" 
output_base_path_label_v = output_base_path + "labels" + "/val"
#本地训练使用
#label_path = "/project/train/src_repo/data_process/temp_label/"
#训练镜像使用
label_path = "/project/train/src_repo/temp_label/"

def getlist(base_path,imglist):
    flist = os.listdir(base_path)
    for f in flist:
        curpath = os.path.join(base_path,f)
        if(os.path.isdir(curpath)):
            getlist(curpath,imglist)
        else:
            imagestr = '.jpg' 
            if(imagestr in f):
                imglist.append(curpath)

def make_data(imglist):
    label_list = os.listdir(label_path)
    rate = 0.8
    for i in range(len(imglist)):
        if(i%10 == 0):
            print(i)
        imgabspath = imglist[i]
        img_name = imgabspath.split('/')[-1]
        label_name = img_name.replace('jpg','txt')
        src_label_path = os.path.join(label_path,label_name)
        #0.7的训练集
        #实际使用的版本
        if(i < len(imglist)*rate):
        #if(True):
        #if(i < 20):
            os.system("ln -s " + imgabspath + " " + output_base_path_img_t + "/" + img_name)
            if(label_name in label_list):
                os.system("ln -s " + " "+ src_label_path + " " + output_base_path_label_t + "/" + label_name) 
        #0.3的验证集
        #实际使用的版本
        else:
        #elif(i< 40):
        #if(True):
            #验证集合复制两次
            os.system("ln -s " + imgabspath + " " + output_base_path_img_t + "/" + img_name)
            if(label_name in label_list):
                os.system("ln -s " + " "+ src_label_path + " " + output_base_path_label_t + "/" + label_name) 
            os.system("ln -s " + imgabspath + " " + output_base_path_img_v + "/" + img_name)
            if(label_name in label_list):
                os.system("ln -s " + " "+ src_label_path + " " + output_base_path_label_v + "/" + label_name) 

if __name__ == "__main__":
    #构建yolo用文件结构
    output_base_path = "~/train/src_repo/dataset/"
    if(not os.path.exists(output_base_path)):
        os.system("mkdir " + output_base_path)
    else:
        os.system("rm -rf " + output_base_path)
    os.system("mkdir " + output_base_path_img)
    os.system("mkdir " + output_base_path_label)
    os.system("mkdir " + output_base_path_img_t)
    os.system("mkdir " + output_base_path_img_v)
    os.system("mkdir " + output_base_path_label_t)
    os.system("mkdir " + output_base_path_label_v)

    #以上构建yolo5需要的文件夹结构,以下填充内容
    in_base = "/home/data/"
    img_list = []
    getlist(in_base,img_list)
    random.shuffle(img_list)
    make_data(img_list)
    #make_data(in_base)
