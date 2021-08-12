import torch
import numpy as np
import os
from sciapp.action import Simple

from imagepy import root_dir
from BulkSeg.model.model import CPnet
from BulkSeg.data_loader.cellpose import util

import albumentations as A
import cv2

def img4torch(image, channel):
    channels= [channel]

    image = np.asarray(image[None,:,:])[channels] if image.ndim==2 else np.asarray(image.transpose(2,0,1))[channels]

    original_size = tuple(image.shape[1:])

    transform = A.Compose([
        A.Resize(height=224, width=224, interpolation=cv2.INTER_NEAREST, p=1)
        ])

    transformed = transform(image=image.transpose(1, 2, 0))
    image = transformed["image"].transpose(2, 0, 1).astype(np.float32)/255

    image_tensor = torch.from_numpy(image).unsqueeze(0)

    return image_tensor, original_size


class Plugin(Simple):
    title = 'Bulk Segmentation'
    note = ['all']
    para = {'weights': '','channel': 0}
    view = [(str, 'weights', 'Weights at', 'directory'), 
            ('lab', 'lab', 'Please select which channel to process. For gray image, just select 0.'),
            (list, 'channel', [0, 1, 2], int, 'channel', 'channel')
    ]

    def run(self, ips, imgs, para = None):

        print("weights_dir = ", para['weights'])

        weights_dir = para['weights']
        weights = torch.load(weights_dir)
        state_dict = weights['state_dict']

        model = CPnet(3, 3)
        model.load_state_dict(state_dict)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        image_tensor, original_size = img4torch(ips.img, para['channel'])

        data = image_tensor.to(device)

        output = model(data)

        flow_prob = output.squeeze(0).cpu().detach().numpy()

        flow_prob = util.resize(flow_prob, original_size)
        flow_prob[2] = util.sigmoid_func(flow_prob[2])
        flow_prob = np.transpose(flow_prob, (1,2,0))

        lab = util.flow2msk(flow_prob)

        self.app.show_img([lab], ips.title+'-mask')
