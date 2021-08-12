import sys
import os.path as osp
bulkseg_path = osp.abspath(osp.dirname(__file__))
print("Bulkseg is at: ", bulkseg_path)
sys.path.append(bulkseg_path)
sys.path.append(osp.join(bulkseg_path, "BulkSeg"))
