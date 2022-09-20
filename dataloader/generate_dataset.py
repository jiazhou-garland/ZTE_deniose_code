import os

def sort_images(image_dir, image_type):
    """
    sort images in the folder based on their names

    image_dir：
    image_type：dng
    """
    files = []
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
                and not image_name.startswith('.'):
            files.append(os.path.join(image_dir, image_name))

    return sorted(files)

def write_file(mode, images, labels):
    """
    save the images and labels into a __.txt file
    """
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(labels)):
            f.write('{}\t{}\t\n'.format(images[i], labels[i]))

def write_file_test(mode, images):
    """
    save the images and labels into a __.txt file
    """
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(images)):
            f.write('{}\n'.format(images[i]))

def train_val_divide(image_dir,label_dir, val_ratio=0.2, fold=0):
    """
    divide the train ana validation data,

    val_ratio = 0.8
    fold = 0
    return: __.txt
            --> Tuple[List[Tuple["image_name","label_name"], List[Tuple["val_image_name","val_label_name"]]]
    """

    print('Start to divide train and val')
    images = sort_images(image_dir, 'dng')
    labels = sort_images(label_dir, 'dng')

    # print(len(images))
    # print(len(labels))
    train_im = []
    train_label = []
    val_im = []
    val_label = []
    val_interval = int((1 / val_ratio))
    for i in range(len(labels)):
        if ((i+1) == fold or (i+1)%val_interval == fold):
            val_im.append(images[i])
            val_label.append(labels[i])
        else:
            train_im.append(images[i])
            train_label.append(labels[i])

    write_file('train', train_im, train_label)
    write_file('val', val_im, val_label)
    print("Done!")

def whole(image_dir,label_dir):

    print('Start to divide train and val')
    images = sort_images(image_dir, 'dng')
    labels = sort_images(label_dir, 'dng')

    # print(len(images))
    # print(len(labels))
    train_im = []
    train_label = []
    for i in range(len(labels)):
        train_im.append(images[i])
        train_label.append(labels[i])

    write_file('whole', train_im, train_label)
    print("Done!")

def test(image_dir):

    images = sort_images(image_dir, 'dng')

    # print(len(images))
    # print(len(labels))
    train_im = []
    for i in range(len(images)):
        train_im.append(images[i])

    write_file_test('test', train_im)
    print("Done!")

if __name__ == "__main__":
    image_dir = '/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/train/ground truth'
    label_dir = '/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/train/noisy'
    test_dir = '/home/zhoujiazhou/PycharmProjects/code/Denoise/dataset/testset'
    # train_val_divide(image_dir, label_dir, val_ratio=0.1, fold=0)
    whole(image_dir, label_dir)
    # test(test_dir)

