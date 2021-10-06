import torch
import numpy as np
import os
from sciapp.action import Simple
from shutil import copy
import zipfile

import fractions
import numpy as np
import torch.nn.functional as F
from torchvision import transforms


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

detransformer = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])


class SingleFaceSwap(Simple):
    title = 'Simple face swapping for already face-aligned images'
    note = ['rgb']
    para = {'name': 'people',
            'Arc_path': 'D:\\src\\weights\\SimSwap-checkpoints\\arcface_checkpoint.tar',
            'insightface_path': 'D:\\src\\weights\\SimSwap-checkpoints\\antelope.zip',
            'gpu_ids': '0',
            'checkpoints_dir': 'D:\\src\\weights\\SimSwap-checkpoints\\checkpoints\\',
            'face_parsing_path': 'D:\\src\\weights\\SimSwap-checkpoints\\79999_iter.pth',
            'model': 'pix2pixHD',
            'norm': 'batch',
            'use_dropout': True,
            'data_type': 32,
            'verbose': False,
            'fp16': False,
            'local_rank': 0,
            'isTrain': False,
            'batchSize': 8,
            'loadSize': 1024,
            'fineSize': 512,
            'label_nc': 0,
            'input_nc': 3,
            'output_nc': 3,
            'resize_or_crop': 'scale_width',
            'serial_batches': True,
            'no_flip': True,
            'nThreads': 2,
            'max_dataset_size': float('inf'),
            'ntest': float('inf'),
            'aspect_ratio': 1.0,
            'phase': 'test',
            'which_epoch': 'latest',
            'des_img': None,
            'use_mask': True
           }
    view = [(str, 'checkpoints_dir', 'Weights file at', 'directory'),
            (str, 'Arc_path', 'Arcface is at', 'directory'),
            (str, 'insightface_path', 'Weights file of insightface is at', 'directory'),
            (str, 'face_parsing_path', 'Weights file of face parsing is at', 'directory'),
            ('img', 'des_img', 'Destination image', '')
    ]

    def run(self, ips, imgs, para = None):
        from models.models import create_model

        start_epoch, epoch_iter = 1, 0
        torch.nn.Module.dump_patches = True

        para = AttrDict(self.para)

        path1 = os.path.abspath(os.path.dirname(__file__))
        face_path = os.path.join(path1, "insightface_func", "models")
        face_parsing_path = os.path.join(path1, "parsing_model", "checkpoint")

        copy(para.insightface_path, face_path)
        copy(para.face_parsing_path, face_parsing_path)

        with zipfile.ZipFile(os.path.join(face_path, 'antelope.zip'), 'r') as zip_ref:
            zip_ref.extractall(face_path)

        model = create_model(para)
        model.eval()


        with torch.no_grad():
            
            img_a = ips.img
            img_a = transformer_Arcface(img_a)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            img_b = self.app.get_img(para.des_img).img
            img_b = transformer(img_b)
            img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()
            img_att = img_att.cuda()

            #create latent id
            img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
            latend_id = model.netArc(img_id_downsample)
            latend_id = latend_id.detach().to('cpu')
            latend_id = latend_id/np.linalg.norm(latend_id,axis=1,keepdims=True)
            latend_id = latend_id.to('cuda')


            ############## Forward Pass ######################
            img_fake = model(img_id, img_att, latend_id, latend_id, True)


            for i in range(img_id.shape[0]):
                if i == 0:
                    row1 = img_id[i]
                    row2 = img_att[i]
                    row3 = img_fake[i]
                else:
                    row1 = torch.cat([row1, img_id[i]], dim=2)
                    row2 = torch.cat([row2, img_att[i]], dim=2)
                    row3 = torch.cat([row3, img_fake[i]], dim=2)

            #full = torch.cat([row1, row2, row3], dim=1).detach()
            full = row3.detach()
            full = full.permute(1, 2, 0)
            output = full.to('cpu')
            output = np.array(output)
            # output = output[..., ::-1]

            output = output*255

        self.app.show_img([output], ips.title+'-swap')


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

class ArbitraryFaceSwap(Simple):
    title = 'Face swapping for Arbitrary images'
    note = ['rgb']
    para = {'name': 'people',
            'Arc_path': 'D:\\src\\weights\\SimSwap-checkpoints\\arcface_checkpoint.tar',
            'insightface_path': 'D:\\src\\weights\\SimSwap-checkpoints\\antelope.zip',
            'gpu_ids': '0',
            'checkpoints_dir': 'D:\\src\\weights\\SimSwap-checkpoints\\checkpoints\\',
            'face_parsing_path': 'D:\\src\\weights\\SimSwap-checkpoints\\79999_iter.pth',
            'model': 'pix2pixHD',
            'norm': 'batch',
            'use_dropout': True,
            'data_type': 32,
            'verbose': False,
            'fp16': False,
            'local_rank': 0,
            'isTrain': False,
            'batchSize': 8,
            'loadSize': 1024,
            'fineSize': 512,
            'label_nc': 0,
            'input_nc': 3,
            'output_nc': 3,
            'resize_or_crop': 'scale_width',
            'serial_batches': True,
            'no_flip': True,
            'nThreads': 2,
            'max_dataset_size': float('inf'),
            'ntest': float('inf'),
            'aspect_ratio': 1.0,
            'phase': 'test',
            'which_epoch': 'latest',
            'des_img': None,
            'use_mask': True
           }
    view = [(str, 'checkpoints_dir', 'Weights file at', 'directory'),
            (str, 'Arc_path', 'Arcface is at', 'directory'),
            (str, 'insightface_path', 'Weights file of insightface is at', 'directory'),
            (str, 'face_parsing_path', 'Weights file of face parsing is at', 'directory'),
            ('img', 'des_img', 'Destination image', '')
    ]

    def run(self, ips, imgs, para = None):
        from models.models import create_model
        from util.norm import SpecificNorm
        from parsing_model.model import BiSeNet
        from util.reverse2original import reverse2wholeimage
        from insightface_func.face_detect_crop_single import Face_detect_crop

        start_epoch, epoch_iter = 1, 0
        crop_size = 224
        torch.nn.Module.dump_patches = True

        para = AttrDict(self.para)

        path1 = os.path.abspath(os.path.dirname(__file__))
        face_path = os.path.join(path1, "insightface_func", "models")
        face_parsing_path = os.path.join(path1, "parsing_model", "checkpoint")

        copy(para.insightface_path, face_path)
        copy(para.face_parsing_path, face_parsing_path)

        with zipfile.ZipFile(os.path.join(face_path, 'antelope.zip'), 'r') as zip_ref:
            zip_ref.extractall(face_path)

        model = create_model(para)
        model.eval()

        spNorm = SpecificNorm()
        app = Face_detect_crop(name='antelope', root=face_path)
        app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))


        with torch.no_grad():
            
            img_a_whole = ips.img
            img_a_align_crop, _ = app.get(img_a_whole, crop_size)
            img_a = transformer_Arcface(img_a_align_crop[0])
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
            # convert numpy to tensor
            img_id = img_id.cuda()
            #create latent id
            img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            img_b_whole = self.app.get_img(para.des_img).img

            img_b_align_crop_list, b_mat_list = app.get(img_b_whole,crop_size)
            # detect_results = None
            swap_result_list = []
            b_align_crop_tenor_list = []


            for b_align_crop in img_b_align_crop_list:
                b_align_crop_tenor = _totensor(b_align_crop)[None,...].cuda()
                swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            if para.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                net.load_state_dict(torch.load(para.face_parsing_path))
                net.eval()
            else:
                net = None


            output = reverse2wholeimage(b_align_crop_tenor_list, swap_result_list, b_mat_list, crop_size, img_b_whole, None, '', True, pasring_model=net, use_mask=para.use_mask, norm=spNorm)

        self.app.show_img([output], ips.title+'-swap')

plgs = [SingleFaceSwap, ArbitraryFaceSwap]