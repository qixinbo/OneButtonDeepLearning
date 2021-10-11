# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from sciapp.action import Simple

from collections import OrderedDict
from torch.autograd import Variable
from Global.models.models import create_model
from Global.models.mapping_model import Pix2PixHDModel_Mapping
from Global.detection_models import networks

import Global.util.util as util
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2
import gc

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def irregular_hole_synthesize(img, mask):
    img_np = img.astype("uint8")
    mask_np = mask.astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = img_new.astype("uint8")

    return hole_img

def detection_scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")

def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    # opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"

class RestoreImage(Simple):
    title = 'Restore Image by Microsoft'
    note = ['rgb']

    para = {'name': 'label2city',
            'gpu_ids': '0',
            'GPU': '0',
            'checkpoints_dir': 'D:\\src\\weights\\oldphotostolife\\global_checkpoints\\checkpoints\\restoration\\',
            'detection_checkpoints_dir': 'D:\\src\\weights\\oldphotostolife\\global_checkpoints\\checkpoints\\detection\\FT_Epoch_latest.pt',
            # 'shape_predictor_dir': 'D:\\src\\weights\\oldphotostolife\\shape_predictor_68_face_landmarks.dat',
            'isTrain': False,
            'spatio_size': 64,
            'model': 'pix2pixHD',
            'norm': 'instance',
            'use_dropout': False,
            'data_type': 32,
            'verbose': False,
            'batchSize': 1,
            'loadSize': 1024,
            'fineSize': 512,
            'label_nc': 35,
            'input_nc': 3,
            'output_nc': 3,
            'resize_or_crop': 'scale_width',
            'serial_batches': False,
            'no_flip': False,
            'nThreads': 2,
            'max_dataset_size': float('inf'),
            'netG': 'global',
            'ngf': 64,
            'k_size': 3,
            'use_v2': False,
            'mc': 1024,
            'start_r': 3,
            'n_downsample_global': 4,
            'n_blocks_global': 9,
            'n_blocks_local': 3,
            'n_local_enhancers': 1,
            'niter_fix_global': 0,
            'load_pretrain': '',
            'no_instance': False,
            'instance_feat': False,
            'label_feat': False,
            'feat_num': 3,
            'load_features': False,
            'n_downsample_E': 4,
            'nef': 16,
            'n_clusters': 10,
            'self_gen': False,
            'map_mc': 64,
            'kl': 0,
            'load_pretrainA': '',
            'load_pretrainB': '',
            'feat_gan': False,
            'no_cgan': False,
            'map_unet': False,
            'map_densenet': False,
            'fcn': False,
            'is_image': False,
            'label_unpair': False,
            'mapping_unpair': False,
            'unpair_w': 1.0,
            'pair_num': -1,
            'Gan_w': 1,
            'feat_dim': -1,
            'abalation_vae_len': -1,
            'use_skip_model': False,
            'use_segmentation_model': False,
            'test_random_crop': False,
            'contain_scratch_L': False,
            'mask_dilation': 0,
            'irregular_mask': '',
            'mapping_net_dilation': 1,
            'non_local': '',
            'NL_fusion_method': 'add',
            'NL_use_mask': False,
            'correlation_renormalize': False,
            'Smooth_L1': False,
            'face_restore_setting': 1,
            'test_on_synthetic': False,
            'use_SN': False,
            'use_two_stage_mapping': False,
            'L1_weight': 10,
            'softmax_temperature': 1.0,
            'patch_similarity': False,
            'use_self': False,
            'use_own_dataset': False,
            'test_hole_two_folders': False,
            'no_hole': False,
            'random_hole': False,
            'NL_res': False,
            'image_L1': False,
            'hole_image_no_mask': False,
            'down_sample_degradation': False,
            'norm_G': 'spectralinstance',
            'init_G': 'xavier',
            'use_new_G': False,
            'use_new_D': False,
            'cosin_similarity': False,
            'downsample_mode': 'nearest',
            'mapping_exp': 0,
            'inference_optimize': False,
            'ntest': float('inf'),
            'aspect_ratio': 1.0,
            'phase': 'test',
            'no_degradation': False,
            'no_load_VAE': False,
            'use_v2_degradation': False,
            'use_vae_which_epoch': 'latest',
            'which_epoch': 'latest',
            'multi_scale_test': 0.5,
            'multi_scale_threshold': 0.5,
            'scale_num': 1,
            'test_mode': 'Full',
            'Quality_restore': True,
            'Scratch_and_Quality_restore': False,
            'with_scratch': False,
            'HR': False,
            'checkpoint_name': 'Setting_9_epoch_100',
            'input_size': 'full_size'
           }
    view = [(str, 'checkpoints_dir', 'Weights file at', 'directory'),
            (str, 'detection_checkpoints_dir', 'Weights of detection at', 'directory'),
            # (str, 'shape_predictor_dir', 'Weights of shape_predictor at', 'directory'),
            (str, 'gpu_ids', 'gpu ids:', 'e.g. 0  0,1,2, 0,2. use -1 for CPU'),
            # (list, 'input_nc', [1, 3], int, 'Input image', 'channels'),
            # (list, 'output_nc', [1, 3], int, 'Output image', 'channels'),
            (list, 'resize_or_crop', ['resize_and_crop', 'crop', 'scale_width', 'scale_width_and_crop'], str, 'scaling and cropping of images', 'when loading'),
            (list, 'data_type', [8,16,32], int, 'Supported data type', 'bit')
            # (list, 'test_mode', ['Scale', 'Full', 'Crop'], str, 'test mode', 'mode')
    ]


    def run(self, ips, imgs, para = None):

        opt = AttrDict(self.para)

        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                opt.gpu_ids.append(int_id)

        parameter_set(opt)

        ## 
        model = Pix2PixHDModel_Mapping()

        model.initialize(opt)
        model.eval()

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        mask_transform = transforms.ToTensor()

        input = ips.img
        input_name = ips.name

        origin = input
        input = img_transform(input)
        input = input.unsqueeze(0)
        mask = torch.zeros_like(input)

        with torch.no_grad():
            generated = model.inference(input, mask)

        output = ((generated.data.cpu()[0] + 1.0) / 2.0).permute(1, 2, 0).numpy()
        print("Finish Stage 1 ...")
        print("\n")

        self.app.show_img([output], ips.title+'-restored')


class RestoreScratchedImage(Simple):
    title = 'Restore Scratched Image by Microsoft'
    note = ['rgb']

    para = {'name': 'label2city',
            'gpu_ids': '0',
            'GPU': '0',
            'checkpoints_dir': 'D:\\src\\weights\\oldphotostolife\\global_checkpoints\\checkpoints\\restoration\\',
            'detection_checkpoints_dir': 'D:\\src\\weights\\oldphotostolife\\global_checkpoints\\checkpoints\\detection\\FT_Epoch_latest.pt',
            # 'shape_predictor_dir': 'D:\\src\\weights\\oldphotostolife\\shape_predictor_68_face_landmarks.dat',
            'isTrain': False,
            'spatio_size': 64,
            'model': 'pix2pixHD',
            'norm': 'instance',
            'use_dropout': False,
            'data_type': 32,
            'verbose': False,
            'batchSize': 1,
            'loadSize': 1024,
            'fineSize': 512,
            'label_nc': 35,
            'input_nc': 3,
            'output_nc': 3,
            'resize_or_crop': 'scale_width',
            'serial_batches': False,
            'no_flip': False,
            'nThreads': 2,
            'max_dataset_size': float('inf'),
            'netG': 'global',
            'ngf': 64,
            'k_size': 3,
            'use_v2': False,
            'mc': 1024,
            'start_r': 3,
            'n_downsample_global': 4,
            'n_blocks_global': 9,
            'n_blocks_local': 3,
            'n_local_enhancers': 1,
            'niter_fix_global': 0,
            'load_pretrain': '',
            'no_instance': False,
            'instance_feat': False,
            'label_feat': False,
            'feat_num': 3,
            'load_features': False,
            'n_downsample_E': 4,
            'nef': 16,
            'n_clusters': 10,
            'self_gen': False,
            'map_mc': 64,
            'kl': 0,
            'load_pretrainA': '',
            'load_pretrainB': '',
            'feat_gan': False,
            'no_cgan': False,
            'map_unet': False,
            'map_densenet': False,
            'fcn': False,
            'is_image': False,
            'label_unpair': False,
            'mapping_unpair': False,
            'unpair_w': 1.0,
            'pair_num': -1,
            'Gan_w': 1,
            'feat_dim': -1,
            'abalation_vae_len': -1,
            'use_skip_model': False,
            'use_segmentation_model': False,
            'test_random_crop': False,
            'contain_scratch_L': False,
            'mask_dilation': 0,
            'irregular_mask': '',
            'mapping_net_dilation': 1,
            'non_local': '',
            'NL_fusion_method': 'add',
            'NL_use_mask': False,
            'correlation_renormalize': False,
            'Smooth_L1': False,
            'face_restore_setting': 1,
            'test_on_synthetic': False,
            'use_SN': False,
            'use_two_stage_mapping': False,
            'L1_weight': 10,
            'softmax_temperature': 1.0,
            'patch_similarity': False,
            'use_self': False,
            'use_own_dataset': False,
            'test_hole_two_folders': False,
            'no_hole': False,
            'random_hole': False,
            'NL_res': False,
            'image_L1': False,
            'hole_image_no_mask': False,
            'down_sample_degradation': False,
            'norm_G': 'spectralinstance',
            'init_G': 'xavier',
            'use_new_G': False,
            'use_new_D': False,
            'cosin_similarity': False,
            'downsample_mode': 'nearest',
            'mapping_exp': 0,
            'inference_optimize': False,
            'ntest': float('inf'),
            'aspect_ratio': 1.0,
            'phase': 'test',
            'no_degradation': False,
            'no_load_VAE': False,
            'use_v2_degradation': False,
            'use_vae_which_epoch': 'latest',
            'which_epoch': 'latest',
            'multi_scale_test': 0.5,
            'multi_scale_threshold': 0.5,
            'scale_num': 1,
            'test_mode': 'Full',
            'Quality_restore': False,
            'Scratch_and_Quality_restore': True,
            'with_scratch': True,
            'HR': False,
            'checkpoint_name': 'Setting_9_epoch_100',
            'input_size': 'full_size',
            'scratch_img': ''
           }
    view = [(str, 'checkpoints_dir', 'Weights file at', 'directory'),
            (str, 'detection_checkpoints_dir', 'Weights of detection at', 'directory'),
            # (str, 'shape_predictor_dir', 'Weights of shape_predictor at', 'directory'),
            (str, 'gpu_ids', 'gpu ids:', 'e.g. 0  0,1,2, 0,2. use -1 for CPU'),
            # (list, 'input_nc', [1, 3], int, 'Input image', 'channels'),
            # (list, 'output_nc', [1, 3], int, 'Output image', 'channels'),
            (list, 'resize_or_crop', ['resize_and_crop', 'crop', 'scale_width', 'scale_width_and_crop'], str, 'scaling and cropping of images', 'when loading'),
            # ('img', 'scratch_img', 'Scratch image', ''),
            (list, 'data_type', [8,16,32], int, 'Supported data type', 'bit')
    ]


    def run(self, ips, imgs, para = None):

        opt = AttrDict(self.para)

        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                opt.gpu_ids.append(int_id)

        parameter_set(opt)

        detect_model = networks.UNet(
            in_channels=1,
            out_channels=1,
            depth=4,
            conv_num=2,
            wf=6,
            padding=True,
            batch_norm=True,
            up_mode="upsample",
            with_tanh=False,
            sync_bn=True,
            antialiasing=True,
        )

        ## load detect_model
        checkpoint = torch.load(opt.detection_checkpoints_dir, map_location="cpu")
        detect_model.load_state_dict(checkpoint["model_state"])
        print("detection model weights loaded")

        print("opt.gpu_ids = ", opt.gpu_ids)

        if len(opt.gpu_ids) > 0:
            detect_model.to(int(opt.gpu_ids[0]))
        else: 
            detect_model.cpu()
        detect_model.eval()


        scratch_image = ips.img
        scratch_image = cv2.cvtColor(scratch_image, cv2.COLOR_BGR2GRAY)

        scratch_image = tv.transforms.ToTensor()(scratch_image)
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        scratch_image = torch.unsqueeze(scratch_image, 0)
        _, _, ow, oh = scratch_image.shape
        scratch_image_scale = detection_scale_tensor(scratch_image)

        if len(opt.gpu_ids) > 0:
            scratch_image_scale = scratch_image_scale.to(int(opt.gpu_ids[0]))
        else:
            scratch_image_scale = scratch_image_scale.cpu()
        with torch.no_grad():
            P = torch.sigmoid(detect_model(scratch_image_scale))

        P = P.data.cpu()
        P = F.interpolate(P, [ow, oh], mode="nearest")

        scratch = (P >= 0.4).float()[0].permute(1, 2, 0).numpy()

        gc.collect()
        torch.cuda.empty_cache()
        self.app.show_img([scratch], ips.title+'-scratch')


        ## 
        model = Pix2PixHDModel_Mapping()

        model.initialize(opt)
        model.eval()

        img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        mask_transform = transforms.ToTensor()

        input = ips.img
        input_name = ips.name


        mask = self.app.get_img(ips.title+'-scratch').img
        if opt.mask_dilation != 0:
            kernel = np.ones((3,3),np.uint8)
            mask = cv2.dilate(mask,kernel,iterations = opt.mask_dilation)
        origin = input

        input = irregular_hole_synthesize(input, mask)
        mask = mask_transform(mask)
        mask = mask[:1, :, :]  ## Convert to single channel
        mask = mask.unsqueeze(0)
        input = img_transform(input)
        input = input.unsqueeze(0)

        with torch.no_grad():
            generated = model.inference(input, mask)

        output = ((generated.data.cpu()[0] + 1.0) / 2.0).permute(1, 2, 0).numpy()
        print("Finish Stage 1 ...")
        print("\n")

        self.app.show_img([output], ips.title+'-stage1')


plgs = [RestoreImage, RestoreScratchedImage]