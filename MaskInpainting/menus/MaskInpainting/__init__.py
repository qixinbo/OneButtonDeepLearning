import sys
import os.path as osp
mask_inpainting_path = osp.abspath(osp.dirname(__file__))
print("mask_inpainting is at: ", mask_inpainting_path)
sys.path.append(mask_inpainting_path)
