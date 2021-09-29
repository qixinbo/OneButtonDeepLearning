import torch
from sciapp.action import Simple
import numpy as np
import cv2

from FBA_Matting.networks.transforms import trimap_transform, normalise_image
from FBA_Matting.networks.models import build_model

def np_to_torch(x, permute=True):
    if permute:
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
    else:
        return torch.from_numpy(x)[None, :, :, :].float().cuda()

def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale

def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    ''' Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    '''
    h, w = trimap_np.shape[:2]
    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():
        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(
            trimap_transform(trimap_scale_np), permute=False)
        image_transformed_torch = normalise_image(
            image_torch.clone())

        output = model(
            image_torch,
            trimap_torch,
            image_transformed_torch,
            trimap_transformed_torch)
        output = cv2.resize(
            output[0].cpu().numpy().transpose(
                (1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)

    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]

    return fg, bg, alpha


class Plugin(Simple):
    title = 'Universal Matting'
    note = ['rgb']
    para = {'weights': '','img':None}
    view = [(str, 'weights', 'Weights at', 'directory'),
            ('img', 'img', 'Trimap image', '')]

    def run(self, ips, imgs, para = None):
        img = (ips.img / 255.0)[:, :, ::-1]

        ips2 = self.app.get_img(para['img'])
        trimap_im = ips2.img
        print("trimap  = ", trimap_im[300, 400])
        h, w = trimap_im.shape
        trimap = np.zeros((h, w, 2))
        trimap[trimap_im == 1, 1] = 1
        trimap[trimap_im == 0, 0] = 1


        model = build_model(para['weights'])
        model.eval().cuda()

        fg, bg, alpha = pred(img, trimap, model)

        self.app.show_img([fg[:, :, ::-1] * 255], ips.title+"foreground")
        self.app.show_img([bg[:, :, ::-1] * 255], ips.title+"background")
        self.app.show_img([alpha*255], ips.title+"alpha")
