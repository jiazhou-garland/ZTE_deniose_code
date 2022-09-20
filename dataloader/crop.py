import os
import numpy as np
import cv2
import rawpy
from os.path import join, abspath, dirname
from scipy import io

def read_image(input_path):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    return raw_data_expand_c, height, width


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[:, :, 2 * channel_y + channel_x]
    return output_data

def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")

def get_image_patch(image_dir, patch_path, image_or_label):

    num = [str(i) for i in range(100)]

    for k in range(100):
        path = image_dir + "/" + image_or_label + '/' + num[k] +  "_gt.dng"
        print(path)

        image, height, width = read_image(path)
        # print(image.shape) # (1736, 2312, 4)
        for h in range(4):
            for w in range(8):

                image_patch = image[h * 256:(h + 2) * 256, w * 256:(w + 2) * 256, :]

                if not os.path.exists(patch_path+image_or_label):
                    os.makedirs(patch_path+image_or_label)
                patch_path = patch_path + "/" + image_or_label + '/' +"h" + str(h+1) + "w" + str(w+1) +  "_gt.dng"
                print(image_patch.shape)
                image_patch = write_image(image_patch, height, width)

                write_back_dng(path, patch_path, image_patch)

                print(patch_path + " has been saved")

if __name__ == '__main__':
    image_dir = '/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/train'
    patch_path = '/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/train_crop'
    get_image_patch(image_dir, patch_path, image_or_label='ground truth')