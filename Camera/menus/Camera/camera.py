import numpy as np
import cv2

class AbsCamera:
    name = 'Abs Camera'
    para, view = {}, None
    
    def __init__(self): self.isopen = False

    def open(self):
        self.isopen = True
        print('camera opened!')

    def close(self):
        self.isopen = False
        print('camera closed!')

    def frame(self, img):
        return np.zeros((512, 512), dtype=np.uint8)

    def config(self, **key):
        print(key)
    
class RandomCamera(AbsCamera):
    name = 'Random Camera'
    
    def __init__(self):
        AbsCamera.__init__(self)
        self.para = {'nframe':10, 'width':512, 'height':512, 'channels':1}
        
        self.view = [(int, 'nframe', (1, 20), 0, 'nframe', 'fps'),
                     (int, 'width', (256, 1024), 0, 'width', 'pix'),
                     (int, 'height', (256, 1024), 0, 'height', 'pix'),
                     (list, 'channels', [1, 3], int, 'channels', 'n')]

    def frame(self):
        from time import sleep
        sleep(1/self.para['nframe'])
        w, h, c = [self.para[i] for i in ['width', 'height', 'channels']]
        return np.random.randint(0, 256, (h, w, c), dtype=np.uint8)

    def config(self, **key): self.para.update(key)

class USBCamera(AbsCamera):
    name = 'USB Camera'
    para_init = True

    def __init__(self):
        AbsCamera.__init__(self)
        self.para = {'port': 0}
        self.view = [(int, 'port', (0, 15), 0, 'com', 'port')]
        
    def open(self):
        self.camera = cv2.VideoCapture(self.para['port'])

    def frame(self):
        return cv2.cvtColor(self.camera.read()[1], cv2.COLOR_BGR2RGB)
    
    def close(self):
        self.camera.release()

            
if __name__ == '__main__':
    import cv2
    camera = RandomCamera()
    camera.config(channels=1)
    while True:
        cv2.imshow('camera', camera.frame())
        cv2.waitKey(1)
