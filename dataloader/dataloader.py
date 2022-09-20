import random, torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import rawpy
from Denoise.Img_denoise.dataloader.data_process import normalization, read_image
import albumentations as A

def train_augmentation():
    return A.Compose([
        A.RandomCrop(width=256, height=256),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
    ])

class CustomImageDataset(Dataset):
    def __init__(self, DataPath, transform=None, black_level = 1024, white_level = 16383, test = False):
        self.DataPath = DataPath
        self.dataList = []
        self.length =self._readTXT(self.DataPath)
        self.random_indices = np.random.permutation(self.length)
        self.test = test
        self.black_level = black_level
        self.white_level = white_level
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        label_path, input_path = self.dataList[idx]
        # print("image:" + input_path)
        # print("label:" + label_path)
        image, height, width = read_image(input_path)
        image = normalization(image, self.black_level, self.white_level)
        # print("image:"+ str(image.shape))  #(1736, 2312, 4)

        label = rawpy.imread(label_path).raw_image_visible
        label = normalization(label, self.black_level, self.white_level)
        label = np.expand_dims(label, axis=2)
        label = np.concatenate((label[0:height:2, 0:width:2, :],
                                label[0:height:2, 1:width:2, :],
                                label[1:height:2, 0:width:2, :],
                                label[1:height:2, 1:width:2, :]), axis=2)
        # print("label:"+str(label.shape))      #label:(1736, 2312, 4)

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        image = torch.from_numpy(np.transpose(image, (0, 3, 1, 2))).float()
        label = torch.from_numpy(np.transpose(label, (0, 3, 1, 2))).float()
        # print("image:"+ str(image.shape))  # image:torch.Size([1, 4, 256, 256])
        # print("label:"+str(label.shape))   # label:torch.Size([1, 4, 256, 256])
        if self.test == True:
            return image, label, input_path, label_path
        else :
            return  image, label

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                image_path, label_path = line.strip().split('\t')
                self.dataList.append((image_path, label_path))
        # random.shuffle(self.dataList)
        return len(self.dataList)

class CustomImageDataset_test(Dataset):
    def __init__(self, DataPath, black_level = 1024, white_level = 16383):
        self.DataPath = DataPath
        self.dataList = []
        self.length =self._readTXT(self.DataPath)
        self.random_indices = np.random.permutation(self.length)
        self.black_level = black_level
        self.white_level = white_level

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        input_path = self.dataList[idx]
        image, height, width = read_image(input_path)
        image = normalization(image, self.black_level, self.white_level)
        # print("image:"+ str(image.shape))  #(1736, 2312, 4)

        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(np.transpose(image, (0, 3, 1, 2))).float()
        # print("image:"+ str(image.shape))  # image:torch.Size([1, 4, 256, 256])

        return image, input_path

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                image_path = line.strip()
                self.dataList.append(image_path)
        return len(self.dataList)

def visualize(input_path,ground_path):
    f0 = rawpy.imread(ground_path)
    f1 = rawpy.imread(input_path)

    plt.subplot(1, 2, 1)
    plt.title('gt')
    plt.imshow(f0.postprocess(use_camera_wb=True))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('noisy')
    plt.imshow(f1.postprocess(use_camera_wb=True))
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    train_path = r'/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/dataloader/train.txt'
    datasets = CustomImageDataset(train_path, test=True)
    feeder = DataLoader(datasets, batch_size=1, shuffle=True)
    # image_path, label_path= next(iter(feeder))
    # print("image:" + image_path[0])
    # print("label:" + label_path[0])
    #
    # image = image.cpu().detach().numpy().squeeze(axis=0).transpose(0, 2, 3, 1)
    # image = inv_normalization(image, black_level = 1024, white_level = 16383)
    #
    # label = label.cpu().detach().numpy().squeeze(axis=0).transpose(0, 2, 3, 1)
    # label = inv_normalization(label, black_level = 1024, white_level = 16383)

    # visualize(image_path[0], label_path[0])

    for step, (batch_x, batch_y) in enumerate(feeder):
        print("image:" + batch_x[0])
        print("label:" + batch_y[0])
        visualize(batch_x[0], batch_y[0])



