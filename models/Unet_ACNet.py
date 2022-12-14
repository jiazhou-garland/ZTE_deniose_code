import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from .ACNet_Block import ACNet_Block

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Unet_ACNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(Unet_ACNet, self).__init__()

        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conv1_1 = ACNet_Block(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = ACNet_Block(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = ACNet_Block(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = ACNet_Block(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = ACNet_Block(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 =ACNet_Block(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = ACNet_Block(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = ACNet_Block(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = ACNet_Block(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = ACNet_Block(512, 512, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6_1 = ACNet_Block(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = ACNet_Block(256, 256, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7_1 = ACNet_Block(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = ACNet_Block(128, 128, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8_1 = ACNet_Block(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2 =ACNet_Block(64, 64, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv9_1 = ACNet_Block(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = ACNet_Block(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = ACNet_Block(32, out_channels, kernel_size=1, stride=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        n, c, h, w = x.shape
        h_pad = 32 - h % 32 if not h % 32 == 0 else 0
        w_pad = 32 - w % 32 if not w % 32 == 0 else 0
        padded_image = F.pad(x, (0, w_pad, 0, h_pad), 'replicate')

        conv1 = self.leaky_relu(self.conv1_1(padded_image))
        conv1 = self.leaky_relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.leaky_relu(self.conv2_1(pool1))
        conv2 = self.leaky_relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)

        conv3 = self.leaky_relu(self.conv3_1(pool2))
        conv3 = self.leaky_relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)

        conv4 = self.leaky_relu(self.conv4_1(pool3))
        conv4 = self.leaky_relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)

        conv5 = self.leaky_relu(self.conv5_1(pool4))
        conv5 = self.leaky_relu(self.conv5_2(conv5))

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.leaky_relu(self.conv6_1(up6))
        conv6 = self.leaky_relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.leaky_relu(self.conv7_1(up7))
        conv7 = self.leaky_relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.leaky_relu(self.conv8_1(up8))
        conv8 = self.leaky_relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.leaky_relu(self.conv9_1(up9))
        conv9 = self.leaky_relu(self.conv9_2(conv9))

        conv10 = self.conv10_1(conv9)
        out = conv10[:, :, :h, :w]

        return out

    def leaky_relu(self, x):
        out = torch.max(0.2 * x, x)
        return out + x


if __name__ == "__main__":
    test_input = torch.from_numpy(np.random.randn(1, 4, 512, 512)).float()
    net = Unet_ACNet()
    output = net(test_input)
    print("test over")