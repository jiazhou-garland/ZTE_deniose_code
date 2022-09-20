import os
import time
import imageio
from skimage import io
import numpy as np
import rawpy
import glob
import argparse
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from scipy.stats import norm
from scipy.optimize import leastsq
import math
from PIL import Image
import cv2

def normalization(input_data, black_level, white_level):
    output_data = np.maximum(input_data.astype(float) - black_level, 0) / (white_level - black_level)
    return output_data

def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
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
    # (3472, 4624, 1) -> (1736, 2312, 4)
    return raw_data_expand_c, height, width

def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[:, :, 2 * channel_y + channel_x]
    # (1736, 2312, 4) -> (3472, 4624, 1)
    return output_data

def denoise_raw(input_path, output_path, black_level, white_level, noiseprofile_a, noiseprofile_b):
    """
    here are how we generate simulated noise profile for your information
    """
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
    # print(raw_data_expand_c_normal.shape)
    raw_data_expand_c_normal_var = noiseprofile_a * raw_data_expand_c_normal + noiseprofile_b

    noise_data = np.random.normal(loc=raw_data_expand_c_normal,
                                  scale=np.sqrt(raw_data_expand_c_normal_var),
                                  size=None)

    noise_data = inv_normalization(noise_data, black_level, white_level)
    noise_data = write_image(noise_data, height, width)
    write_back_dng(input_path, output_path, noise_data)

def err(p, x, y):
    return p[0]*x + p[1] - y

def cal_noise_profile(path_rgb, noise_path, args):
    """
    your code should be given here
    """
    # img, h, w = read_image(test_dir)
    # img = normalization(img, black_level, white_level)

    RGB_image = io.imread(path_rgb)
    # print(RGB_image.shape)
    start1 = time.time()
    segments = slic_segments(RGB_image, args)
    end1 = time.time()
    print(f'SLIC segmentation lasted time: {end1 - start1}')
    # print(segments)
    set_segments = list(set(np.reshape(segments, segments.size)))
    print('set_segments:' + str(len(set_segments)))

    mean_list = []
    sigma_list = []
    sigma_statistical_list = []
    start2 = time.time()
    img, _, _ = read_image(noise_path)
    img = normalization(img, args.black_level, args.white_level)
    # print(img.shape)

    x = np.arange(0,0.25,0.01)
    y = args.noiseprofile_a*x + args.noiseprofile_b

    y_alow_blow = args.noiseprofile_a_range_low*x + args.noiseprofile_b_range_low
    y_ahigh_blow = args.noiseprofile_a_range_high*x + args.noiseprofile_b_range_low
    y_alow_bhigh = args.noiseprofile_a_range_low*x + args.noiseprofile_b_range_high
    y_ahigh_bhigh = args.noiseprofile_a_range_high*x + args.noiseprofile_b_range_high

    for i in range(4):
        for j in range(len(set_segments)):
            tmp = img[:,:,i]
            tmp = tmp[np.where(segments == set_segments[j])]
            mean_statistical = np.mean(tmp)
            sigma_statistical = np.var(tmp)
            sigma_statistical_list.append(sigma_statistical)
            if sigma_statistical > 0.03:
                continue
            # a = tmp.size
            tmp = tmp[np.logical_and(tmp > 0, tmp < 1)] # decrease the influence of np.clip
            # a = tmp.size
            tmp = tmp[np.logical_and(tmp > mean_statistical-3*sigma_statistical, tmp < mean_statistical + 3*sigma_statistical)] #three sigma law
            # b = tmp.size
            # print(a-b)
            mean, sigma = fused_maxinum_likelyhood(tmp)
            # print(f'mean: {mean:.12f} sigma: {sigma:.12f} mean_statistical:'
            #       f' {mean_statistical:.12f} sigma_statistical: {sigma_statistical:.12f}')

            sigma_high_high = args.noiseprofile_a_range_high*mean + args.noiseprofile_b_range_high - sigma*sigma
            sigma_low_low = args.noiseprofile_a_range_low*mean + args.noiseprofile_b_range_low - sigma*sigma
            if sigma_high_high > 0 and sigma_low_low < 0: # the limits of parameters a and b
                mean_list.append(mean)
                sigma_list.append(sigma*sigma)
            # print(mean)
            # print(sigma)
    print("The amount of points:" + str(len(mean_list)))
    plt.hist(sigma_statistical_list)
    plt.show()

    plt.scatter(mean_list, sigma_list, marker='x', c='red')
    plt.plot(x,y, c='yellow')
    plt.plot(x,y_alow_blow,c='blue')
    plt.plot(x,y_ahigh_blow,c='blue')
    plt.plot(x,y_alow_bhigh,c='blue')
    plt.plot(x,y_ahigh_bhigh,c='blue')
    plt.show()

    P0 = [0.001, 0.0001]
    (a,b), _ = leastsq(err, P0, args=(np.array(mean_list), np.array(sigma_list)))
    # (a, b) = np.polyfit(np.array(mean_list), np.array(sigma_list), deg=1)
    end2 = time.time()
    print(f'leastsq lasted time: {end2 - start2}')

    if b > args.noiseprofile_b_range_high:
        b = min(b, args.noiseprofile_b_range_high)
    if b < args.noiseprofile_b_range_low:
        b = max(b, args.noiseprofile_b_range_low)
    if a > args.noiseprofile_a_range_high :
        a = min(b, args.noiseprofile_a_range_high)
    if a < args.noiseprofile_a_range_low:
        a = max(b + args.noiseprofile_a_range_low)

    print(f'The predicted a is {a:.12f} and b is {b:.12f}')
    print(f'The amount of time for predicting one image : {end2 - start2 + end1 - start1}')
    print('-----------------------------------------------')

    return a,b

def data_analysis(args):
    # input_dir = args.input_dir # gt
    input_dir = args.output_dir  # noise_image
    path = glob.glob(input_dir + '*.dng')
    for index in range(len(path)):
        in_path = path[index]
        in_basename = os.path.basename(in_path)
        input_path = input_dir + in_basename
        print(input_path)

        img, h, w = read_image(input_path)
        img = normalization(img, args.black_level, args.white_level)
        img_data = np.reshape(img,img.size)
        # print(img_data.shape)
        # print(max(img_data)-min(img_data))

        d = 0.0001
        num_bins = int((max(img_data)-min(img_data)) // d)
        # print(num_bins)

        # 设置图形大小
        plt.figure(figsize=(20, 8), dpi=80)
        plt.hist(img_data, num_bins)

        # 设置x轴刻度
        # plt.xticks(range(min(img_data), max(img_data) + d, d))

        # 设置网格
        plt.grid(alpha=0.4)
        plt.show()

def fused_maxinum_likelyhood(x):

    (mean, sigma) = norm.fit(x)
    return mean, sigma

def slic_segments(img, args):

    numSegments = args.nums_segment
    sigma = args.sigma
    # image：待执行SLTC超像素分割的图像
    # n_segments: 定义我们要生成多少个超像素段的参数，默认100
    # sigma：在分割之前应用的平滑高斯核
    segments = slic(img, n_segments=numSegments, sigma=sigma, start_label=1, compactness=10)

    return segments

def visualize(args):
    input_dir = args.input_dir  # noise_image
    output_dir = args.output_dir
    path_gt = glob.glob(input_dir + '*.dng')
    path_noise = glob.glob(output_dir + '*.dng')

    for index in range(len(path_gt)):
        gt_path = path_gt[index]
        gt_basename = os.path.basename(gt_path)
        input_path = input_dir + gt_basename
        noise_path = path_noise[index]
        noise_basename = os.path.basename(noise_path)
        output_path = output_dir + noise_basename
        print(output_path)
        print(input_path)

        f0 = rawpy.imread(input_path)
        f1 = rawpy.imread(output_path)

        plt.subplot(1, 2, 1)
        plt.title('gt')
        plt.imshow(f0.postprocess(use_camera_wb=True))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('noisy')
        plt.imshow(f1.postprocess(use_camera_wb=True))
        plt.axis('off')

        plt.show()

def visualize_slic(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    rgb_dir = args.save_jpg_dir
    path_gt = glob.glob(input_dir + '*.dng')
    path_noise = glob.glob(output_dir + '*.dng')
    path_rgb = glob.glob(rgb_dir + '*.jpeg')

    for index in range(len(path_rgb)):
        # gt_path = path_gt[index]
        # gt_basename = os.path.basename(gt_path)
        # input_path = input_dir + gt_basename
        # # print(input_path)
        # noise_path = path_noise[index]
        # noise_basename = os.path.basename(noise_path)
        # output_path = output_dir + noise_basename
        # # print(output_path)

        rgb_path = path_rgb[index]
        print(rgb_path)
        RGB_image = io.imread(rgb_path)
        print(RGB_image.shape)

        # f1 = rawpy.imread(input_path)
        # f2 = rawpy.imread(output_path)
        # start = time.time()
        segments = slic_segments(RGB_image, args)
        #
        # end = time.time()
        # print(f'lasted time: {end-start}')
        print(segments)
        # set_segments = list(set(np.reshape(segments,segments.size)))
        # print(len(set_segments))
        #
        fig = plt.figure("Superpixels -- %d segments" % (args.nums_segment))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(RGB_image, segments, color=(1, 1, 0), mode='thick'))
        plt.axis("off")
        #
        # plt.subplot(1, 3, 2)
        # plt.title('rgb')
        # plt.imshow(f1.postprocess(use_camera_wb=True))
        # plt.axis('off')
        #
        # plt.subplot(1, 3, 3)
        # plt.title('noisy')
        # plt.imshow(f2.postprocess(use_camera_wb=True))
        # plt.axis('off')
        #
        plt.show()

def calculate_frequency(args):
    input_dir = args.output_dir  # noise_image
    path = glob.glob(input_dir + '*.dng')
    for index in range(len(path)):
        in_path = path[index]
        in_basename = os.path.basename(in_path)
        input_path = input_dir + in_basename
        img, h, w = read_image(input_path)
        img = normalization(img, args.black_level, args.white_level)
        img_data = np.reshape(img,img.size)
        img_data = sorted(img_data)
        img_data_set = list(set(img_data))
        # print(img_data)
        # print(img_data.size)
        # print(len(img_data_set))

        d = 0.001
        num_bins = int((max(img_data) - min(img_data)) // d)
        freq = np.empty(num_bins, dtype=np.int64)

        for i in range(len(img_data)):
            for j in range(num_bins):
                if j*d <= img_data[i] < (j+1)*d:
                    freq[j] += 1
        print(freq)

def save_jpg_image(img_dir,save_dir):
    # dir = args.output_dir # noise_image
    # save_dir = args.save_jpg_dir
    path = glob.glob(img_dir + '*.dng')

    for index in range(len(path)):
        path_one = path[index]
        basename = os.path.basename(path_one)
        input_path = img_dir + basename
        save_path = save_dir + basename[:-4] + '.jpeg'
        # print(save_path)

        with rawpy.imread(input_path) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
            # print(rgb.shape)
        imageio.imsave(save_path, rgb)
        rgb = cv2.imread(save_path)
        rgb = cv2.resize(rgb, (2312, 1736))
        # print(rgb.shape)
        cv2.imwrite(save_path,rgb)

def main(args):
    # TODO the codes below for offline train!!!
    input_dir = args.input_dir
    output_dir = args.output_dir
    save_jpg_dir = args.save_jpg_dir
    if not os.path.exists(save_jpg_dir):
        os.mkdir(save_jpg_dir)

    #TODO Release the codes below for validation and note the above codes!!!
    # output_dir = args.test_dir
    # save_jpg_dir = args.save_test_jpg_dir
    # if not os.path.exists(save_jpg_dir):
    #     os.mkdir(save_jpg_dir)

    black_level = args.black_level
    white_level = args.white_level
    noiseprofile_a = args.noiseprofile_a
    noiseprofile_b = args.noiseprofile_b

    score = []

    for idx in range(len(noiseprofile_a)):
        """
        this part is an example showing how to generate simulated noise
        you do not need to modify this part
        """
        print('-----------------------------------------------')
        path = glob.glob(input_dir + '*.dng')
        for index in range(len(path)):
            in_path = path[index]
            in_basename = os.path.basename(in_path)
            input_path = input_dir + in_basename
            out_basename = in_basename.split(".")[0].strip()
            output_path = output_dir + out_basename + "_noise" + ".dng"
            denoise_raw(input_path, output_path, black_level, white_level, noiseprofile_a[idx], noiseprofile_b[idx])

        """
        this part aims to test your algorithm performance
        we will use multiple images and generate multiple noise profile para. to test your result
        three images in ./data/test/ is an example to help you understand our evaluation criteria
        you will not see ground truth test image and corresponding noise profile para. in test proc. 
        """
        # save_jpg_image(output_dir, save_jpg_dir)
        path_rgb = glob.glob(save_jpg_dir + '*.jpeg')
        path_noise = glob.glob(output_dir + '*.dng')
        for index in range(len(path_rgb)):
            rgb_path = path_rgb[index]
            noise_path = path_noise[index]
            print("rgb_path " + str(rgb_path))
            print("path_noise " + str(noise_path))

            """modify your function cal_noise_profile"""
            a, b = cal_noise_profile(rgb_path, noise_path, args)

            evm_tmp = 0.5 * np.min([np.abs(a - noiseprofile_a[idx]) / noiseprofile_a[idx], 1]) \
                      + 0.5 * np.min([np.abs(b - noiseprofile_b[idx]) / noiseprofile_b[idx], 1])
            score_tmp = 100 - 100 * evm_tmp

            score.append(score_tmp)

    """your final score"""
    print('each score =', score)
    print('final score =', np.mean(score))

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./data/gt/")
    parser.add_argument('--output_dir', type=str, default="./data/noise/")
    parser.add_argument('--test_dir', type=str, default="./data/test/")
    parser.add_argument('--save_jpg_dir', type=str, default="./data/jpg/")
    parser.add_argument('--save_test_jpg_dir', type=str, default="./data/test_rgb/")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--noiseprofile_a', type=float, default=[0.005])
    parser.add_argument('--noiseprofile_b', type=float, default=[0.0005])
    parser.add_argument('--nums_segment', type=float, default=900)
    parser.add_argument('--sigma', type=int, default=20)
    parser.add_argument('--noiseprofile_a_range_low', type=float, default=1e-4)
    parser.add_argument('--noiseprofile_a_range_high', type=float, default=1e-2)
    parser.add_argument('--noiseprofile_b_range_low', type=float, default=1e-6)
    parser.add_argument('--noiseprofile_b_range_high', type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
    end = time.time()
    print(f'total lasted time: {end-start}')

    # data_analysis(args)
    # calculate_frequency(args)
    # visualize(args)
    # visualize_slic(args)
    # cal_noise_profile(args.output_dir, args.save_jpg_dir, args)