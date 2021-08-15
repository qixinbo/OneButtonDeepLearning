import sys
import os.path as osp
face_path = osp.abspath(osp.dirname(__file__))
print("face analysis is at: ", face_path)
sys.path.append(face_path)
