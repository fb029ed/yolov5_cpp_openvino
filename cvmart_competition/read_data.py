import os
import xml.etree.ElementTree as et
import cv2

def getlist(base_path,labels_list,images_list):
    flist = os.listdir(base_path)
    for f in flist:
        curpath = os.path.join(base_path,f)
        if(os.path.isdir(curpath)):
            getlist(curpath,labels_list,images_list)
        else:
            labelstr = '.xml'
            imagestr = '.jpg' 
            if(labelstr in f):
                labels_list.append(curpath)
            elif(imagestr in f):
                images_list.append(curpath)

def voc2coco(labels_list,images_list):
    output_base_path = "/project/train/src_repo/temp_label"
    if(not os.path.exists(output_base_path)):
        os.system("mkdir " + output_base_path)
    need_write_list = []
    for xmlf in labels_list:
        img_path = xmlf.replace('.xml','.jpg')   #abs_path
        #img_path = os.path.join(base_path,img_path)
        if(not(img_path in images_list)):
            continue
        img = cv2.imread(img_path)
        size = img.shape
        #注意检查此处序号是否正确
        w = int(size[1])
        h = int(size[0])   
        tree = et.parse(xmlf)
        root = tree.getroot()  
        class_name = '0' 
        troot = root.find('object')
        if(None == troot):
            continue
        need_copy = False
        small_th = 0.03
        children_nodes = root.getchildren() 
        #遍历子节点加入box
        temp = ""
        for child in children_nodes: 
            if(child.tag == "object"):
                xmin = child.find('bndbox').find('xmin').text
                ymin = child.find('bndbox').find('ymin').text
                xmax = child.find('bndbox').find('xmax').text
                ymax = child.find('bndbox').find('ymax').text
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                xcenter_rate = float((xmax + xmin)/2) / w
                ycenter_rate = float((ymax + ymin)/2) / h
                rate_w = float(xmax - xmin) / w
                rate_h = float(ymax - ymin) / h
                temp += class_name + " " + str(xcenter_rate) + " " + str(ycenter_rate) + " " + str(rate_w) + " " + str(rate_h) + "\n"
                if(rate_h < small_th  or rate_w  < small_th):
                    need_copy = True
        temp_label_name = output_base_path+ '/' + xmlf.split('/')[-1].replace('xml','txt')
        with open(temp_label_name,mode='w') as f:
            f.write(temp)
            f.close()
        #需要增强的数据进行复制
        #TODO:增加复制次数
        if(need_copy):
            #cp img(ln实现)
            src_path = img_path #绝对路径
            tar_path = img_path.split(".")[-2] + "_2.jpg"
            #print("copy path :" + tar_path)
            os.system("ln -s " + " "+ src_path + " " +tar_path) 
            #cp txt
            temp_label_name_2 = output_base_path+ '/' + xmlf.split('/')[-1].split(".")[-2] + "_2.txt"
            with open(temp_label_name_2,mode='w') as f:
                f.write(temp)
                f.close()

if __name__ == '__main__':
    in_basepath = "/home/data/"
    labels_list = []
    images_list = []
    getlist(in_basepath,labels_list,images_list)
    print("size of label" + str(len(labels_list)))
    print("size of image" + str(len(images_list)))
    voc2coco(labels_list,images_list)

