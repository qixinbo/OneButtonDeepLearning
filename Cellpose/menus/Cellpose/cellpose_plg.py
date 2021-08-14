import numpy as np
import pandas as pd
from sciapp.action import Simple

class Plugin(Simple):
    title = 'Cellpose'
    note = ['all']
    para = {'model':'cyto', 'cytoplasm':0, 'nucleus':0, 'diameter': 30, 'flow':False, 'overlay': False, 'diams':False, 'slice':False}
    view = [(list, 'model', ['cyto', 'nuclei'], str, 'model', ''),
            (list, 'cytoplasm', [0,1,2,3], int, 'cytoplasm', 'channel'),
            (list, 'nucleus', [0,1,2,3], int, 'nucleus', 'channel'),
            (int, 'diameter', (5, 1000), 30, 'diameter', 'pixels'),
            (bool, 'flow', 'show color flow'),
            (bool, 'overlay', 'show overlay'),
            (bool, 'diams', 'show diams tabel'),
            (bool, 'slice', 'slice')]

    def run(self, ips, imgs, para = None):
        from cellpose import models, utils
        from cellpose.plot import mask_overlay

        use_GPU = models.use_gpu()

        if not para['slice']: imgs = [ips.img]
        imgs = [i.reshape((i.shape+(-1,))[:3]) for i in imgs]

        model = models.Cellpose(gpu=use_GPU, model_type=para['model'])
        channels = [para['cytoplasm'], para['nucleus']]
        self.setValue = lambda x: self.progress(x, 100)
        masks, flows, styles, diams = model.eval(
            imgs, diameter= para['diameter'], rescale=None, channels=channels, progress=self)
        self.app.show_img(masks, ips.title+'-cp_mask')
        if para['flow']: self.app.show_img([i[0] for i in flows], ips.title+'-cp_flow')
        if para['overlay']:
            for i in range(len(imgs)):
                maski = masks[i].copy().astype(np.uint8)
                imgi = imgs[i].copy()
                overlayi = mask_overlay(imgi, maski)
                self.app.show_img([overlayi], ips.title+'-overlay')
        if para['diams']: 
            self.app.show_table(pd.DataFrame({'diams': diams}), ips.title+'-cp_diams')