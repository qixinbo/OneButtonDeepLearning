from sciapp.action import Simple
from sciapp.util import mark2shp
import os
import torch
import numpy as np

from insightface.app import FaceAnalysis

class Plugin(Simple):
    title = 'Face Recognition'
    note = ['rgb']
    para = {'ctx': 0, 'size': 640}
    view = [
    (int, 'ctx', (-1, 10), 0, 'ctx id', '<0 means using cpu'),
    (int, 'size', (0, 5120), 640, 'detection size', 'pixels')
    ]

    def run(self, ips, imgs, para=None):
        app = FaceAnalysis()
        app.prepare(ctx_id=para['ctx'], det_size=(int(para['size']), int(para['size'])))
        faces = app.get(ips.img)

        mark = {'type': 'layer', 'body': []}

        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)

            rect = {'type': 'rectangle',
            'body': (box[0], box[1], box[2]-box[0], box[3]-box[1]),
            'lw': 3,
            'color': color
            }

            mark['body'].append(rect)

            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)

                    circle = {'type': 'circle',
                    'body': (kps[l][0], kps[l][1], 1),
                    'lw': 2,
                    'color': color}
                    mark['body'].append(circle)

            # print("face.gender = ", face.gender)
            # if face.gender is not None:
            #     text = {'type':'text',
            #         'body': (box[0]-1, box[1]-4, 'Gender is %s'%(face.gender)), 
            #         'pt':False, 
            #         'lw': 1,
            #         'color': (0,255,0)}

            #     mark['body'].append(text)
            if face.age is not None:
                text = {'type':'text',
                    'body': (box[0]-1, box[1]-4, 'Age = %d'%(face.age)), 
                    'pt':False, 
                    'lw': 8,
                    'color': (0,255,0)}

                mark['body'].append(text)

        ips.mark = mark2shp(mark)
        ips.update()