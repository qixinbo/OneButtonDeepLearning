import torch
from torchvision import transforms
import numpy as np
import os
from sciapp.action import Simple
from imagepy import root_dir

class Plugin(Simple):
    title = 'DeepLab V3'
    note = ['rgb']
    # para = {'weights': '','channel': 0}
    # view = [(str, 'weights', 'Weights at', 'directory'), 
    #         ('lab', 'lab', 'Please select which channel to process. For gray image, just select 0.'),
    #         (list, 'channel', [0, 1, 2], int, 'channel', 'channel')
    # ]

    def run(self, ips, imgs, para = None):
        model = torch.hub.load('pytorch/vision:v0.8.2', 'deeplabv3_resnet101', pretrained=True)
        model.eval()

        preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(ips.img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]

        output_predictions = output.argmax(0).cpu().numpy()
        # print("output.shape = ", output_predictions.shape)

        self.app.show_img([output_predictions], ips.title + '-seg') 