import paddle
import numpy as np
import cv2
from paddle.vision.models import resnet50
from paddle.vision.datasets import DatasetFolder
import paddle.nn.functional as F
import matplotlib.pylab as plt
import os
import random
import shutil
from shutil import copy2

train_file='train'
valid_file='valid'
test_file='test'
imagesize=32
batch_size=32
lr=1e-5
iters=10000


def data_set_split(src_data_folder, target_data_folder, train_scale=0.8, val_scale=0.1, test_scale=0.1):
    '''
    读取源数据文件夹，生成划分好的文件夹，分为trian、val、test三个文件夹进行
    :param src_data_folder: 源文件夹
    :param target_data_folder: 目标文件夹
    :param train_scale: 训练集比例
    :param val_scale: 验证集比例
    :param test_scale: 测试集比例
    :return:
    '''
    print("开始数据集划分")
    class_names = os.listdir(src_data_folder)
    # 在目标目录下创建文件夹
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)
        # 然后在split_path的目录下创建类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行数据图片的复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        if os.path.isdir(current_class_data_path):
            current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, 'train'), class_name)
        val_folder = os.path.join(os.path.join(target_data_folder, 'val'), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, 'test'), class_name)
        train_stop_flag = current_data_length * train_scale
        val_stop_flag = current_data_length * (train_scale + val_scale)
        current_idx = 0
        train_num = 0
        val_num = 0
        test_num = 0
        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                try:
                    copy2(src_img_path, train_folder)
                    # print("{}复制到了{}".format(src_img_path, train_folder))
                    train_num = train_num + 1
                except:
                    pass
            else:
                try:
                    copy2(src_img_path, val_folder)
                    # print("{}复制到了{}".format(src_img_path, val_folder))
                    val_num = val_num + 1
                except:
                    pass

            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(class_name))
        print(
            "{}类按照{}：{}：{}的比例划分完成，一共{}张图片".format(class_name, train_scale, val_scale, test_scale,
                                                                 current_data_length))
        print("训练集{}：{}张".format(train_folder, train_num))
        print("验证集{}：{}张".format(val_folder, val_num))


if __name__ == '__main__':
    src_data_folder = "smoking and calling image_datasets/train"
    target_data_folder = "work"
    data_set_split(src_data_folder, target_data_folder)

# 定义数据预处理
def load_image(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imagesize, imagesize))
    img = np.array(img).astype('float32') / 255.0  # 归一化到 [0, 1]
    img = img.transpose((2, 0, 1))  # 转换为 CHW
    return img

# 构建Dataset
class Face(DatasetFolder):
    def __init__(self, path):
        super().__init__(path)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        label = np.array(label).astype(np.int64)
        label = np.expand_dims(label, axis=0)
        return load_image(img_path), label


train_file = "work/train"
valid_file = "work/val"

train_dataset = Face(train_file)
eval_dataset = Face(valid_file)
# 定义数据预处理
def load_image_1(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = np.array(img).astype('float32')
    # HWC to CHW
    img = img.transpose((2,0,1))
    #Normalize
    img = img / 255
    return img
# 构建Dataset
class Face_1(DatasetFolder):
    def __init__(self, path):
        super().__init__(path)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        label = np.array(label).astype(np.int64)
        label = np.expand_dims(label, axis=0)
        return load_image_1(img_path), label

train_file = "work/train"
valid_file = "work/val"

train_dataset_1 = Face_1(train_file)
eval_dataset_1 = Face_1(valid_file)

# 显示图像
plt.figure(figsize=(8, 6))  # 根据需要调整 figure 的大小
for i in range(10):
    fundus_img, lab = train_dataset.__getitem__(i)

    # 将图像从 CHW 转换为 HWC 格式以进行显示
    fundus_img = fundus_img.transpose(1, 2, 0)  # 转换为 HxWxC

    # 可选：调整图像尺寸
    #fundus_img = cv2.resize(fundus_img, (128, 128), interpolation=cv2.INTER_CUBIC)

    plt.subplot(2, 5, i + 1)
    plt.imshow(fundus_img, interpolation='bicubic')  # 使用双三次插值
    plt.axis("off")

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()