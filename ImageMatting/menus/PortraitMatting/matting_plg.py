import torch
from RobustVideoMatting.model import MattingNetwork
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from RobustVideoMatting.inference_utils import ImageReader
from sciapp.action import Simple
import numpy as np

class Plugin(Simple):
    title = 'Image Matting'
    note = ['all']

    def run(self, ips, imgs, para = None):
        import os.path as osp
        file_path = osp.join(osp.abspath(osp.dirname(__file__)), ".\\RobustVideoMatting\\rvm_mobilenetv3.pth")

        model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
        model.load_state_dict(torch.load(file_path))

        reader = ImageReader(ips.img, transform=ToTensor())

        bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).cuda()  # Green background.
        rec = [None] * 1                                      # Initial recurrent states.
        downsample_ratio = 0.25                                # Adjust based on your video.

        with torch.no_grad():
            for src in DataLoader(reader):                     # RGB tensor normalized to 0 ~ 1.
                fgr, pha, *rec = model(src.cuda())  # Cycle the recurrent states.
                com = fgr * pha + bgr * (1 - pha)              # Composite to green background. 
                out = com.cpu().numpy()[0].transpose((1, 2, 0))*255
                self.app.show_img([out.astype(np.uint8)], ips.title+'matting')                          # Write frame.