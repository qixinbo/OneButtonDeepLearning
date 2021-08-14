import sys
import os.path as osp
cellpose_path = osp.abspath(osp.dirname(__file__))
print("cellpose is at: ", cellpose_path)
sys.path.append(cellpose_path)