import sys
import os.path as osp
u2net_path = osp.abspath(osp.dirname(__file__))
print("u2net is at: ", u2net_path)
sys.path.append(u2net_path)
