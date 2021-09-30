from sciapp.action import Simple

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2

class Plugin(Simple):
    title = 'Image Restoration'
    note = ['rgb']
    para = {'weights': '','netscale': 4, 'outscale': 4, 'tile': 0, 'tile_pad': 10, 'pre_pad': 0, 'block': 23, 'face_enhance': True, 'half': True, 'AutoDownloadGFPGAN': True, 'GFPGAN_model': 'please fill in the model path from your local directory'}
    view = [(str, 'weights', 'Weights at', 'directory'), 
            (int, 'netscale', (1,10), 0, 'Upsample scale factor', 'of the network'),
            (float, 'outscale', (1,10), 1, 'Final upsampling scale', 'of the image'),
            (int, 'tile', (0,10), 0, 'Tile size', '0 for no tile during testing'),
            (int, 'tile_pad', (0,100), 0, 'Tile padding', 'default is 10'),
            (int, 'pre_pad', (0,100), 0, 'Pre padding size', 'at each border'),
            (int, 'block', (0,100), 0, 'num_block', 'in RRDB'),
            (bool, 'face_enhance', 'Use GFPGAN to enhance face'),
            (bool, 'half', 'Use half precision during inference'),
            (bool, 'AutoDownloadGFPGAN', 'Auto downloading GFPGAN model weights from Github? (about 332MB), or just input the model path below from your local directory'),
            (str, 'GFPGAN_model', 'GFPGAN model weights at', 'directory')
    ]

    def run(self, ips, imgs, para = None):
        if 'RealESRGAN_x4plus_anime_6B.pth' in para['weights']:
            para['block'] = 6
        elif 'RealESRGAN_x2plus.pth' in para['weights']:
            para['netscale'] = 2

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=para['block'], num_grow_ch=32, scale=para['netscale'])


        upsampler = RealESRGANer(
            scale=para['netscale'],
            model_path=para['weights'],
            model=model,
            tile=para['tile'],
            tile_pad=para['tile_pad'],
            pre_pad=para['pre_pad'],
            half=para['half'])

        if para['face_enhance']:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth' if para['AutoDownloadGFPGAN'] else para['GFPGAN_model'],
                upscale=para['outscale'],
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)

        img = ips.img
        h, w = img.shape[0:2]
        if max(h, w) > 1000 and para['netscale'] == 4:
            self.app.alert('The input image is large, try X2 model for better performance.')
        if max(h, w) < 500 and para['netscale'] == 2:
            self.app.alert('The input image is small, try X4 model for better performance.')


        try:
            if para['face_enhance']:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=para['outscale'])

            self.app.show_img([output], ips.title+'-restoration')

        except Exception as error:
            self.app.alert('Error!! If you encounter CUDA out of memory, try to set --tile with a smaller number.')
