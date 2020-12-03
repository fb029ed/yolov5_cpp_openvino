import os 

def find_target_exp():
    base = "/project/"
    flist = os.listdir(base)
    run_path = ""
    for f in flist:
        run_name = "runs" 
        if(os.path.isdir(f) and run_name in f):
            run_path = os.path.join(base,f)
    #src_repo文件夹找不到的时候在yolo文件夹查找
    if(run_path == ""):
        base ="/project/train/src_repo/yolov5/"
        flist = os.listdir(base)
        for f in flist: 
            if(os.path.isdir(f) and run_name in f):
                run_path = os.path.join(base,f)
    if(run_path == ""):
        print("runs not find")
    
    print("find runs path " + run_path)

    #find lastest exp file
    exp_list = os.listdir(run_path)
    max_num = -1
    needed_exp_path = ""
    print("num")
    for exp in exp_list:
        print(exp)
        num = int(exp.split("exp")[-1])
        print(num)
        if(num > max_num):
            max_num = num
            needed_exp_path = exp
    needed_exp_path = os.path.join(run_path,needed_exp_path)
    print("find exp path " + needed_exp_path)
    
    #进行文件的复制
    #权重文件复制
    os.system("mkdir -p /project/train/models/final/")
    pt_path = needed_exp_path + "/weights/best.pt"
    pt_target_path =  "/project/train/models/final/best.pt"
    os.system("cp "+ pt_path + " " + pt_target_path)
    #模型先转换后复制
    os.system("cd /project/train/src_repo/yolov5/")
    os.system('export PYTHONPATH="/project/train/src_repo/yolov5/" ')
    os.system("python3 /project/train/src_repo/yolov5/models/export.py --weights "+pt_target_path+" --img 640 --batch 1")
    #完成onnx转换
    os.system("cp /project/train/src_repo/yolov5/best.onnx /project/train/models/final/best.onnx")
    
    #log文件复制
    logp = needed_exp_path + "/results.txt"
    fr = open("/project/train/log/log.txt","a")
    fo = open(logp,"r")
    lines = fo.read()
    print("\n\n",file = fr)
    print(lines,file=fr)
    fo.close()
    fr.close()
    #resutl img path
    rmp = needed_exp_path + "/results.png"
    result_img_path = "/project/train/result-graphs/results.png"
    os.system("cp " + rmp + " " + result_img_path)
    
if __name__ == '__main__':
    find_target_exp()
