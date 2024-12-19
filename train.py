import paddle
import numpy as np
import cv2
from paddle.vision.models import resnet50
import paddle.nn.functional as F
import matplotlib.pyplot as plt
from paddle.vision.datasets import DatasetFolder

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
def train(model, iters, optimizer, criterion, train_dataset, eval_dataset, log_interval, eval_interval):
    iter = 0
    losses = 5
    model.train()
    avg_loss_list = []
    while iter < iters :
        iter += 1
        for batch_id, train_datas in enumerate(train_loader()):
            train_data, train_label = train_datas
            train_data = paddle.to_tensor(train_data)
            train_label = paddle.to_tensor(train_label)
            ligits = model(train_data)
            loss = criterion(ligits, train_label)
            avg_loss_list.append(loss)
            loss.backward()
            optimizer.step()
            model.clear_gradients()
        if iter % log_interval == 0:
            avg_loss = np.array(avg_loss_list).mean()
            print("[TRAIN] iter={}/{} loss={:.4f}".format(iter, iters, avg_loss))
        if iter % eval_interval == 0:
            model.eval()
            avg_loss_list = []
            for eval_datas in eval_dataset:
                eval_data, eval_label = eval_datas
                eval_data = np.expand_dims(eval_data, axis=0)
                eval_data = paddle.to_tensor(eval_data)
                eval_label = paddle.to_tensor(eval_label)
                ligits = model(eval_data)
                loss = criterion(ligits, eval_label)
                avg_loss_list.append(loss)
            avg_loss = np.array(avg_loss_list).mean()
            print("[EVAL] iter={}/{} loss={:.4f}".format(iter, iters, avg_loss))
            if loss < losses:
                paddle.save(model.state_dict(),os.path.join("best_model_{:.4f}".format(avg_loss), 'model.pdparams'))
            model.train()

train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
optimizer = paddle.optimizer.Adam(lr,  parameters = model.parameters())
criterion = paddle.nn.CrossEntropyLoss()
train(model, iters, optimizer, criterion, train_dataset, eval_dataset, log_interval = 10, eval_interval=100)