import sys
import os.path as osp
imagematting_path = osp.abspath(osp.dirname(__file__))
print("imagematting is at: ", imagematting_path)
sys.path.append(imagematting_path)