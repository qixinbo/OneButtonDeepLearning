import sys
import os.path as osp
faceswap_path = osp.abspath(osp.dirname(__file__))
print("faceswap is at: ", faceswap_path)
sys.path.append(faceswap_path)
