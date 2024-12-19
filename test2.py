import paddle
import cv2
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from paddle.vision.models import resnet50
import paddle.nn.functional as F


train_file = 'train'
valid_file = 'valid'
test_file = 'test'
imagesize = 32
batch_size = 32
lr = 1e-5
iters = 10000


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


# 加载预训练模型
model = Network()
best_model_path = "best_model/model.pdparams"
para_state_dict = paddle.load(best_model_path)
model.set_state_dict(para_state_dict)
model.eval()

# 定义异常行为字典
face = {0: 'calling', 1: 'normal', 2: 'smoking'}


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)

        # OpenCV视频流
        self.video_source = video_source
        self.vid = None

        # 创建一个画布用于显示视频帧
        self.canvas_width = 800  # 设置画布宽度
        self.canvas_height = 600  # 设置画布高度
        self.canvas = tk.Canvas(window, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(padx=10, pady=10)

        # 创建按钮
        self.btn_open_camera = tk.Button(window, text="打开摄像头", width=50, command=self.open_camera)
        self.btn_open_camera.pack(anchor=tk.CENTER, expand=True)

        self.btn_open_file = tk.Button(window, text="打开视频文件", width=50, command=self.open_file)
        self.btn_open_file.pack(anchor=tk.CENTER, expand=True)

        # 显示时间戳和行为类型的标签
        self.label_status = tk.Label(window, text="", fg="red")
        self.label_status.pack()

        # 更新视频帧
        self.delay = 15  # 毫秒
        self.update()

        self.window.mainloop()

    def open_camera(self):
        if self.vid is not None:
            self.vid.release()
        self.vid = cv2.VideoCapture(self.video_source)
        self.update()

    def open_file(self):
        path = filedialog.askopenfilename()
        if path:
            if self.vid is not None:
                self.vid.release()
            self.vid = cv2.VideoCapture(path)
            self.update()

    def update(self):
        # 检查视频流是否打开
        if self.vid is not None and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # 直接使用原始帧，无需任何处理
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 调用模型进行预测
                frame_resized = cv2.resize(frame, (imagesize, imagesize))  # 可根据需要调整大小
                frame_resized = np.array(frame_resized).astype('float32')
                frame_resized = frame_resized.transpose((2, 0, 1))  # 转换为 C x H x W 格式
                frame_resized = frame_resized / 255  # 归一化
                frame_resized = np.expand_dims(frame_resized, axis=0)  # 增加 batch 维度
                frame_resized = paddle.to_tensor(frame_resized)

                with paddle.no_grad():
                    result = model(frame_resized)
                    result = paddle.nn.functional.softmax(result)
                    result = paddle.argmax(result, axis=1).numpy()[0]
                    behavior = face[result]

                # 显示当前时间戳和行为类型
                self.label_status.config(text=f"时间戳: {timestamp}, 行为: {behavior}")

                # 将视频帧显示在画布中
                frame_resized = np.uint8(frame_resized.numpy()[0].transpose(1, 2, 0) * 255)  # 转换为 uint8 类型
                height, width, _ = frame_resized.shape

                # 调整大小以适应画布
                height, width, _ = frame.shape
                width_ratio = self.canvas_width / width
                height_ratio = self.canvas_height / height
                ratio = min(width_ratio, height_ratio)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


                # 显示视频帧
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
                self.canvas.delete("all")
                self.canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=self.photo, anchor=tk.CENTER)

        # 每次延迟后继续更新视频帧
        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.vid is not None:
            self.vid.release()


# 创建GUI窗口
App(tk.Tk(), "PaddlePaddle 视频流")

# 释放资源
if __name__ == '__main__':
    pass
