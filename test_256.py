import torch
from torch.utils.data import DataLoader
from dataloader.dataloader import CustomImageDataset_test
from Denoise.Img_denoise.models.Unet_ACNet import Unet_ACNet
from Denoise.Img_denoise.models.unetTorch import Unet
from models.NAF_model.archs.NAFNet_arch import NAFNet
from matplotlib import pyplot as plt
import rawpy

import torchvision.transforms as T
from dataloader.data_process import write_back_dng, inv_normalization, write_image, read_image

def pred_256(model, batch_x):
    pred_img = torch.zeros(1, 4, 1736, 2312)
    for h in range(13):
        for w in range(17):
            image_patch = batch_x[:, :, h * 128:(h + 2) * 128, w * 128 : (w + 2) * 128]
            pred_img[:, :, h * 128:(h + 2) * 128, w * 128:(w + 2) * 128] = model(image_patch)
    for h in range(13):
        image_patch = batch_x[:, :, h * 128:(h + 2) * 128, 2184:2312]
        pred_img[:, :, h * 128:(h + 2) * 128, 2184:2312] = model(image_patch)
    for w in range(17):
        image_patch = batch_x[:, :, 1608:1736, w * 128:(w + 2) * 128]
        pred_img[:, :, 1608:1736, w * 128:(w + 2) * 128] = model(image_patch)

    image_patch = batch_x[:, :, 1608:1736, 2184:2312]
    pred_img[:, :, 1608:1736, 2184:2312] = model(image_patch)

    return pred_img

if __name__ == '__main__':
    # Prepare model-----------------------------------------------------------------------------------------------------------
    device_num = 2
    device = torch.device("cuda:" + str(device_num))
    img_channel = 4
    width = 36
    enc_blks = [2, 4, 8, 8]
    middle_blk_num = 8
    dec_blks = [2, 4, 8, 8]
    model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    model.to(device)
    model.eval()

    # Load model------------------------------------------------------------------------------------------------------
    model_ckpt = "/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/runs/NAFNet2488_50.87/models/best_model.pth"
    model.load_state_dict(torch.load(model_ckpt))

    # dataloader
    testTxtPath = r'/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/dataloader/test.txt'
    datasets_test = CustomImageDataset_test(testTxtPath)
    test_loader = DataLoader(datasets_test, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
                            drop_last=False, num_workers=1)
    _, height, width = read_image("/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/train/ground truth/0_gt.dng")
    with torch.no_grad():
        for step, (batch_x, input_path) in  enumerate(test_loader):
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x= batch_x.to(device)
            batch_x = batch_x.squeeze(dim=1) # torch.Size([1, 4, 1736, 2312])

            hflipper = T.RandomHorizontalFlip(p=1)
            vflipper = T.RandomVerticalFlip(p=1)

            pred_img1 = pred_256(model, batch_x)

            pred_img2 = pred_256(model, hflipper(batch_x))
            pred_img2 = hflipper(pred_img2)

            pred_img3 = pred_256(model, vflipper(batch_x))
            pred_img3 = vflipper(pred_img3)

            pred_img4 = pred_256(model, vflipper(hflipper(batch_x)))
            pred_img4 = hflipper(vflipper(pred_img4))

            pred_img5 = pred_256(model, hflipper(vflipper(batch_x)))
            pred_img5 = vflipper(hflipper(pred_img5))

            # pred_img1 = model(batch_x)
            #
            # pred_img2 = model(hflipper(batch_x))
            # pred_img2 = hflipper(pred_img2)
            #
            # pred_img3 = model(vflipper(batch_x))
            # pred_img3 = vflipper(pred_img3)
            #
            # pred_img4 = model(vflipper(hflipper(batch_x)))
            # pred_img4 = hflipper(vflipper(pred_img4))
            #
            # pred_img5 = model(hflipper(vflipper(batch_x)))
            # pred_img5 = vflipper(hflipper(pred_img5))

            pred_img = (pred_img1 + pred_img2 + pred_img3 + pred_img4 + pred_img5) / 5

            # postprocess
            result_data = pred_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
            result_data = inv_normalization(result_data,  black_level = 1024, white_level = 16383)
            result_write_data = write_image(result_data, height, width)
            output_path ="/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/result/data/denoise" + str(step) + ".dng"
            write_back_dng(input_path[0], output_path, result_write_data)

            # f = rawpy.imread(output_path)
            # plt.title('predicted_output')
            # plt.imshow(f.postprocess(use_camera_wb=True))
            # plt.axis('off')
            # plt.show()