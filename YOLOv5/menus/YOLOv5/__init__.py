import sys
import os.path as osp
yolo_path = osp.abspath(osp.dirname(__file__))
print("yolo is at: ", yolo_path)
sys.path.append(yolo_path)
# sys.path.append(osp.join(yolo_path, 'ultralytics_yolov5_master'))
