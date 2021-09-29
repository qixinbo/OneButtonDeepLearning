import sys
import os.path as osp
universalmatting_path = osp.abspath(osp.dirname(__file__))
print("universalmatting is at: ", universalmatting_path)
sys.path.append(universalmatting_path)
