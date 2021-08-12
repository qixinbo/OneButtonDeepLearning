import sys
sys.path.append("..")


from torchvision import datasets, transforms
from base import BaseDataLoader
from skimage.io import imread

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


from torch.utils.data.dataset import Dataset
from data_loader.cellpose import util
import numpy as np
import os
import tifffile

class CellposeDataset(Dataset):
    def __init__(self, data_dir, channel=0, transform=None):
        self.channels = [channel]
        self.transform = transform
        self.data_dir = data_dir

        self.image_names = []
        self.mask_names = []

        self.image_dir = os.path.join(self.data_dir, "images")
        self.image_names = list(sorted(os.listdir(self.image_dir)))

        print("self.image_names = ", self.image_names)

        self.mask_dir = os.path.join(self.data_dir, "masks")
        self.mask_names = [i.replace("img", "mask") for i in self.image_names]

        print("self.mask_names = ", self.mask_names)

        self.flow_names = [i.replace("img.png", "flow.tiff") for i in self.image_names]

        print("self.flow_names = ", self.flow_names)


    def __getitem__(self, i):
        image = imread(os.path.join(self.image_dir, self.image_names[i]))
        image = np.asarray(image[None,:,:])[self.channels] if image.ndim==2 else np.asarray(image.transpose(2,0,1))[self.channels]
        mask = imread(os.path.join(self.mask_dir, self.mask_names[i]))

        flow_full_name = os.path.join(self.mask_dir, self.flow_names[i])

        if os.path.exists(flow_full_name):
            flow = imread(flow_full_name).transpose(2, 0 ,1)
        # if 0:
        #     pass
        else:
            # #!!!!!!!!!!!!!!!!!Don't forget to delete generated files
            flow = util.msk2flow(mask)

            # from data_loader.cellpose.others import dynamics            
            # flow = dynamics.masks_to_flows(mask)[0].transpose(1, 2, 0)

            flow = np.concatenate((flow.transpose(2, 0, 1), mask[None, :, :]>0.5), axis=0).astype(np.float32)
            tifffile.imsave(flow_full_name, flow)


        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0), mask=flow.transpose(1, 2, 0))
            image = transformed["image"].transpose(2, 0, 1)
            flow = transformed["mask"].transpose(2, 0, 1)

        # print("image.shape = ", image.shape)
        # print("flow.shape = ", flow.shape)

        return image.astype(np.float32)/255, flow

    def __len__(self):
        return len(self.image_names)


class CellposeInferenceDataset(Dataset):
    def __init__(self, data_dir, channel=0, transform=None):
        self.channels = [channel]
        self.transform = transform
        self.data_dir = data_dir

        self.image_names = []

        self.image_dir = os.path.join(self.data_dir, "images")
        self.image_names = list(sorted(os.listdir(self.image_dir)))

        print("self.image_names = ", self.image_names)

    def __getitem__(self, i):
        image = imread(os.path.join(self.image_dir, self.image_names[i]))

        image = np.asarray(image[None,:,:])[self.channels] if image.ndim==2 else np.asarray(image.transpose(2,0,1))[self.channels]

        original_size = tuple(image.shape[1:])

        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0))
            image = transformed["image"].transpose(2, 0, 1)

        return image.astype(np.float32)/255, original_size

    def __len__(self):
        return len(self.image_names)





import albumentations as A
import cv2

transform = A.Compose([
    A.Resize(height=224, width=224, interpolation=cv2.INTER_NEAREST, p=1)
    ])

class CellposeDataLoader(BaseDataLoader):
    def __init__(self, data_dir, training=True, channel=0, batch_size=128, shuffle=False, validation_split=0.0, num_workers=1, with_transform=True):
        if with_transform:
            self.transform = transform
        else:
            self.transform = None

        if training:
            dataset = CellposeDataset(data_dir, channel, self.transform)
        else:
            dataset = CellposeInferenceDataset(data_dir, channel, self.transform)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':

    cp_loader = CellposeDataLoader(data_dir=r'../data/cellpose/samples', channel=1, batch_size=1, with_transform=False)
    # cp_loader = CellposeDataLoader(data_dir=r'D:\temp\demo', channel=0, batch_size=1, with_transform=False)
    # cp_loader = CellposeInferenceDataset(data_dir='../data/cellpose/samples', transform = transform)
    print(len(cp_loader))
    np.set_printoptions(threshold=np.inf)
    for i, data in enumerate(cp_loader):
        # print(data[0].shape)
        # print(data[1].shape)
        # print(data)
        print("--------------------- " + str(i))
        from data_loader.cellpose import render, util
        import matplotlib.pyplot as plt

        img = data[0][0].permute((1, 2, 0))[:, :, 0].numpy()*255.
        img = img.astype(np.uint8)
        plt.subplot(131).imshow(img)

        flow_prob = data[1][0].numpy()
        flow_prob[2] = util.sigmoid_func(flow_prob[2])
        flow_prob = np.transpose(flow_prob, (1,2,0))


        np.save(str(i)+"-flow.npy", flow_prob)
        np.save(str(i)+"-flow_x.npy", flow_prob[:, :, 0])
        np.save(str(i)+"-flow_y.npy", flow_prob[:, :, 1])

        mask = util.flow2msk(flow_prob)

        from skimage.io import imsave
        np.save(str(i)+"-mask.npy", mask)

        rgb = render.rgb_mask(img, mask)
        plt.subplot(132).imshow(mask*255)

        flow = render.flow2hsv(flow_prob[:, :, :-1])
        plt.subplot(133).imshow(flow)


        # print(flow_prob[:, :, 1].dtype)


        # print("flow_prob[:, :, 0] = ", flow_prob[:, :, 0])
        # print("flow_prob[:, :, 1] = ", flow_prob[:, :, 1])
        # print("flow_prob[:, :, 2] = ", flow_prob[:, :, 2])

        # imsave(str(i)+"-x-flow.png", flow_prob[:, :, 0])
        # imsave(str(i)+"-y-flow.png", flow_prob[:, :, 1])

        plt.show()
        print("****")
