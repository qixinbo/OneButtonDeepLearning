from sciapp.action import Simple
from sciapp.util import mark2shp
import os
import torch
import pandas as pd


class Plugin(Simple):
    title = 'Paddle OCR'
    note = ['all']
    para = {'lang': 'ch'}
    view = [
    (list, 'lang', ['ch', 'en', 'french', 'german', 'korean', 'japan'], str, 'Select', 'language')
    ]

    def run(self, ips, imgs, para=None):
        from paddleocr import PaddleOCR, draw_ocr
        ocr = PaddleOCR(use_angle_cls=True, lang=para['lang']) # need to run only once to download and load model into memory
        result = ocr.ocr(ips.img, cls=True)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        mark = {'type':'layer', 'body':[]}

        for i in range(len(boxes)):
            polygon = {
            'type': 'polygon', 
            'color': (255,0,0), 
            'lw': 1,
            'style':'o',
            'body':boxes[i]}

            mark['body'].append(polygon)

        ips.mark = mark2shp(mark)
        ips.update()

        titles = ['text', 'confidence']

        list_of_tuples = list(zip(txts, scores))
        self.app.show_table(pd.DataFrame(list_of_tuples, columns=titles), ips.title+'-ocr')
