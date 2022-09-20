import torch
from openpyxl import Workbook
import rawpy
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataloader.dataloader import CustomImageDataset
from Denoise.Img_denoise.models.unetTorch import Unet
from models.NAF_model.archs.NAFNet_arch import NAFNet
from dataloader.data_process import write_back_dng, inv_normalization, write_image, read_image
from Denoise.Img_denoise.utils.metics import metics

def write_file(mode, images, labels):
    """
    save the images and labels into a __.txt file
    """
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(labels)):
            f.write('{}\t{}\t\n'.format(images[i], labels[i]))

if __name__ == '__main__':
    wb = Workbook()  # 创建工作簿
    ws = wb.active  # 激活工作表
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
    testTxtPath = r'/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/dataloader/whole.txt'
    datasets_test = CustomImageDataset(testTxtPath, test = True)
    test_loader = DataLoader(datasets_test, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
                            drop_last=False, num_workers=1)
    _, height, width = read_image("/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/train/ground truth/0_gt.dng")
    im45_50 = []
    label45_50 = []
    im40_45 = []
    label40_45 = []
    im_40 = []
    label_40 = []
    with torch.no_grad():
        for step, (batch_x, batch_y, input_path, label_path) in  enumerate(test_loader):
            batch_x = batch_x.type(torch.FloatTensor)
            batch_x= batch_x.to(device)
            batch_x = batch_x.squeeze(dim=1) # torch.Size([1, 4, 1736, 2312])
            pred_img = torch.zeros(1, 4, 1736, 2312)

            pred_img = model(batch_x)
            batch_y = batch_y.squeeze(dim=1)
            psnr, ssim = metics(pred_img, batch_y, 1736, 2312)
            # print(f"psnr: {psnr.item()}")
            # print(f"ssim: {ssim.item()}")
            list_one = [input_path[0]]

            list_one.append(psnr.item())
            list_one.append(ssim.item())
            print(list_one)
            # ws.append(list_one)

            if psnr.item() < 50 and psnr.item() >= 45:
                im45_50.append(input_path[0])
                label45_50.append(label_path[0])
            if (psnr.item() < 45 and psnr.item() >= 40):
                im40_45.append(input_path[0])
                label40_45.append(label_path[0])
            if psnr.item() < 40:
                im_40.append(input_path[0])
                label_40.append(label_path[0])

            # postprocess
            # result_data = pred_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
            # result_data = inv_normalization(result_data,  black_level = 1024, white_level = 16383)
            # result_write_data = write_image(result_data, height, width)
            # output_path ="/home/zhoujiazhou/PycharmProjects/code/Denoise/Img_denoise/result_train/" + input_path[0][67:] + ".dng"
            # write_back_dng(input_path[0], output_path, result_write_data)

            # visuliazation
            # f = rawpy.imread(output_path)
            # plt.title('predicted_output')
            # plt.imshow(f.postprocess(use_camera_wb=True))
            # plt.axis('off')
            # plt.show()

    # wb.save('whole_data_result.xlsx')
    write_file('hard45_50', label45_50, im45_50)
    write_file('hard40_45', label40_45, im40_45)
    write_file('hard40', label_40, im_40)