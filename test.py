import paddle
import numpy as np
import cv2
from paddle.vision.models import resnet50
import paddle.nn.functional as F
import matplotlib.pyplot as plt
from paddle.vision.datasets import DatasetFolder
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

class Network(paddle.nn.Layer):
    def __init__(self):
        super(Network, self).__init__()
        self.resnet = resnet50(pretrained=True, num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear_1 = paddle.nn.Linear(2048, 512)
        self.linear_2 = paddle.nn.Linear(512, 256)
        self.linear_3 = paddle.nn.Linear(256, 7)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(0.2)

    def forward(self, inputs):
        # print('input', inputs)
        y = self.resnet(inputs)
        y = self.flatten(y)
        y = self.linear_1(y)
        y = self.linear_2(y)
        y = self.dropout(y)
        y = self.relu(y)
        y = self.linear_3(y)
        y = paddle.nn.functional.sigmoid(y)
        y = F.softmax(y)

        return y

model = Network()
def load_test(img_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #resize
    img = cv2.resize(img,(imagesize,imagesize))
    img = np.array(img).astype('float32')
    # HWC to CHW
    img = img.transpose((2,0,1))
    #Normalize
    img = img / 255

    return img

test_dataset=[]
for i in os.listdir(test_file):
    test_dataset.append(load_test(test_file+'//'+i))
test_dataset = np.array(test_dataset)
# 进行预测操作
best_model_path = "best_model/model.pdparams"
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()

result_list=[]
for test_data in test_dataset:
    test_data = np.expand_dims(test_data, axis=0)
    test_data = paddle.to_tensor(test_data)
    result = model(test_data)
    result = paddle.tolist(result)
    result_list.append(result)
result_list = np.array(result_list)

# 定义产出数字与行为的对应关系
face={0:'calling',1:'normal',2:'smoking'}

# 定义画图方法
def show_img(img, predict):
    plt.figure()
    plt.title('predict: {}'.format(face[predict]))
    plt.imshow(img.reshape([3, 32, 32]).transpose(1,2,0))
    plt.show()


# 抽样展示
indexs = [89, 18, 78]

for idx in indexs:
    show_img(test_dataset[idx], np.argmax(result_list[idx]))