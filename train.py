python3 /project/train/src_repo/data/read_data.py
python3 /project/train/src_repo/data/make_dataset.py
#python3 /project/train/src_repo/yolov5/train.py --batch 16 --epochs 100 --data /project/train/src_repo/rat.yaml --cfg /project/train/src_repo/yolov5/models/yolov5s.yaml --weights "/project/train/src_repo/yolov5spre.pt" --noautoanchor
python3 /project/train/src_repo/yolov5/train.py --batch 16 --epochs 50 --data /project/train/src_repo/rat.yaml --cfg /project/train/src_repo/yolov5/models/yolov5m.yaml --weights "/project/train/src_repo/yolov5mpre.pt" --noautoanchor
export PYTHONPATH="/project/train/src_repo/yolov5/"
python3 /project/train/src_repo/data/save_data.py
