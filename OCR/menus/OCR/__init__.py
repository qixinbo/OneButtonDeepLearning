import sys
import os.path as osp
ocr_path = osp.abspath(osp.dirname(__file__))
print("ocr is at: ", ocr_path)
sys.path.append(ocr_path)
sys.path.append(osp.join(ocr_path, 'PaddleOCR-2-1-1'))