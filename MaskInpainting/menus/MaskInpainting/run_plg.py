import torch
import numpy as np
import os
from sciapp.action import Simple

from lama.saicinpainting.evaluation.utils import move_to_device

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from lama.saicinpainting.training.data.datasets import make_default_val_dataset
from lama.saicinpainting.training.trainers import load_checkpoint

from torch.utils.data import Dataset
def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod
def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')

class InpaintingDataset(Dataset):
    def __init__(self, img, mask, pad_out_to_modulo=8, scale_factor=None):
        self.img = img
        self.mask = mask
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return 1 

    def __getitem__(self, i):
        image = self.img 
        mask = self.mask
        result = dict(image=image, mask=mask[None, ...])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class Plugin(Simple):
    title = 'Image Inpainting'
    note = ['rgb']
    para = {'model_path': 'D:\\src\\weights\\big-lama\\', 
            'mask': '',
            'device': 'cuda'}
    view = [(str, 'model_path', 'Model is at', 'directory'),
            ('img', 'mask', 'Mask image', ''),
            (str, 'device', 'device', 'cuda or cpu')
    ]

    def run(self, ips, imgs, para = None):
        img_origin = ips.img
        img = np.transpose(img_origin, (2, 0, 1)).astype('float32') / 255
        mask_origin = self.app.get_img(para['mask']).img
        mask = mask_origin.astype('float32') / 255

        device = torch.device(para['device'])

        train_config_path = os.path.join(para['model_path'], 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(para['model_path'], 
                                       'models', 
                                       'best.ckpt')
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)

        dataset = InpaintingDataset(img, mask)
        with torch.no_grad():
            for img_i in range(len(dataset)):
                batch = move_to_device(default_collate([dataset[img_i]]), device)
                batch['mask'] = (batch['mask'] > 0) * 1
                batch = model(batch)
                cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                self.app.show_img([cur_res], ips.title+"-inpainting")
