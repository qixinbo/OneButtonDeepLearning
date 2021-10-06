import os
from sciapp.action import Simple
from skimage import io, transform
from skimage.filters import gaussian
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from U2Net.data_loader import RescaleT
from U2Net.data_loader import ToTensor
from U2Net.data_loader import ToTensorLab
from U2Net.data_loader import SalObjDataset

from U2Net.model import U2NET # full size version 173.6 MB
from U2Net.model import U2NETP # small version u2net 4.7 MB

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

class SOD(Simple):
    title = 'Salient Object Detection'
    note = ['rgb']
    para = {'weights': 'D:\\src\\weights\\u2net.pth'}
    view = [(str, 'weights', 'Weights at', 'directory')
    ]

    def run(self, ips, imgs, para = None):
        model_name = 'u2net'

        test_salobj_dataset = SalObjDataset(img_name_list = [ips.img],
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)


        # --------- 3. model define ---------
        model_dir = para['weights']
        if(model_name=='u2net'):
            print("...load U2NET---173.6 MB")
            net = U2NET(3,1)
        elif(model_name=='u2netp'):
            print("...load U2NEP---4.7 MB")
            net = U2NETP(3,1)

        if torch.cuda.is_available():
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()

        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

            # normalization
            pred = d1[:,0,:,:]
            predict = normPRED(pred)
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()

            im = Image.fromarray(predict_np*255).convert('RGB')
            image = ips.img
            imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

            pb_np = np.array(imo)
            del d1,d2,d3,d4,d5,d6,d7

            self.app.show_img([pb_np], ips.title+'-mask')


class Portrait(Simple):
    title = 'Portrait Generation'
    note = ['rgb']
    para = {'weights': 'D:\\src\\weights\\u2net_portrait.pth',
            'sigma': 20,
            'alpha': 0.5,
            'useGPU': False}
    view = [(str, 'weights', 'Weights at', 'directory'),
            (int, 'sigma', (0, 100), 0, 'sigma of gaussian function', 'for blurring the orignal image'),
            (float, 'alpha', (0, 1), 2, 'alpha weights of the orignal image', 'when fusing portrait'),
            (bool, 'useGPU', 'Use GPU')
    ]

    def run(self, ips, imgs, para = None):
        model_name = 'u2net_portrait'

        test_salobj_dataset = SalObjDataset(img_name_list = [ips.img],
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(512),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)


        # --------- 3. model define ---------
        model_dir = para['weights']
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)

        if self.para['useGPU']:
            net.load_state_dict(torch.load(model_dir))
            net.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        net.eval()

        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):
            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if self.para['useGPU']:
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)

            d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

            # normalization
            pred = 1.0 - d1[:,0,:,:]
            predict = normPRED(pred)
            predict = predict.squeeze()
            predict_np = predict.cpu().data.numpy()
            image = ips.img

            pd = transform.resize(predict_np,image.shape[0:2],order=2)
            pd = pd/(np.amax(pd)+1e-8)*255
            pd = pd[:,:,np.newaxis]

            ## fuse the orignal portrait image and the portraits into one composite image
            ## 1. use gaussian filter to blur the orginal image
            sigma=float(self.para['sigma'])
            image = gaussian(image, sigma=sigma, preserve_range=True)

            ## 2. fuse these orignal image and the portrait with certain weight: alpha
            alpha = float(self.para['alpha'])
            im_comp = image*alpha+pd*(1-alpha)

            del d1,d2,d3,d4,d5,d6,d7

            self.app.show_img([im_comp], ips.title+'-composite')

plgs = [SOD, Portrait]