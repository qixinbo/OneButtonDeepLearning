from sciapp.action import Simple
from sciapp.util import mark2shp
import os
import torch

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors() 

class Plugin(Simple):
	title = 'YOLOv5 detection'
	note = ['all']
	para = {'model': 'yolov5s'}
	view = [
	('lab', 'lab', 'Choose the specific YOLOv5 model:'),
	('lab', 'lab', 'yolov5s for small version, m for medium version, l for large, x for xlarge'),
	(list, 'model', ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'], str, 'Select', 'model')
	]

	def run(self, ips, imgs, para=None):
		dirname = os.path.dirname(__file__)
		model_dir = os.path.join(dirname, 'ultralytics_yolov5_master')
		model = torch.hub.load(model_dir, para['model'], pretrained=True, source='local')
		result = model(ips.img)
		print("pandas = ", result.pandas().xyxy[0])

		mark = {'type':'layer', 'body':[]}

		for i, (im, pred) in enumerate(zip(result.imgs, result.pred)):
			if pred.shape[0]:
				for *box, conf, cls in reversed(pred):
					label = f'{result.names[int(cls)]} {conf:.2f}'

					rect = {'type':'rectangle',
					'body':(int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])),
					'lw': 3,
					'color':colors(cls)}

					text = {'type':'text',
					'body':(int(box[0]), int(box[1])-12, label), 
					'pt':False, 
					'lw': 8,
					'fill': True,
					'color': (255, 255, 255),
					'fcolor':colors(cls)}

					mark['body'].append(rect)
					mark['body'].append(text)

		ips.mark = mark2shp(mark)
		ips.update()