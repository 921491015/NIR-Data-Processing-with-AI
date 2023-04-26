import numpy as np
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 绘图必须添加这个，否则出对象错误
from sklearn.preprocessing import scale, MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

test_ratio = 0.2  # 测试集比例
EPOCH = 700  # 训练次数
BATCH_SIZE = 256  # 每次训练使用的样本数
LR = 0.0005  # 学习率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_result_path = r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\CNN\train_result.csv'
test_result_path = r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\CNN\test_result.csv'

grape_data = pd.read_csv(
    r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
y = grape_data.iloc[:, 100:101].values  # 标签
y = np.array(y)
print(y.dtype)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
grape_data = grape_data.iloc[:, :100]
grape_array = np.array(grape_data)
x = grape_array
print(torch.cuda.is_available())
data_x = x
data_y = y

x_lenth = len(data_x[1, :])
print(x_lenth)

x_data = np.array(data_x)
y_data = np.array(data_y)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15)
'''
X_train = x_data
X_test = x_data
y_train = y_data
y_test = y_data
'''

##自定义加载数据集
class MyDataset(Dataset):
    def __init__(self, specs, labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec, target = self.specs[index], self.labels[index]
        return spec.astype(np.float32), target

    def __len__(self):
        return len(self.specs)


##均一化处理
X_train_Nom = scale(X_train)
X_test_Nom = scale(X_test)
X_train_Nom = X_train_Nom[:, np.newaxis, :]
X_test_Nom = X_test_Nom[:, np.newaxis, :]
data_train = MyDataset(X_train_Nom, y_train)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)

##使用loader加载测试数据
data_test = MyDataset(X_test_Nom, y_test)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=400, shuffle=True)


class NIR_CONV3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(NIR_CONV3, self).__init__()
        self.CONV1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm1d(output_channel),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.CONV2 = nn.Sequential(
            nn.Conv1d(output_channel, 83, 25, 1, 1),
            nn.BatchNorm1d(83),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.CONV3 = nn.Sequential(
            nn.Conv1d(83, 32, 2, 1),
            nn.BatchNorm1d(32),  # 对输出做均一化
            nn.ReLU(),
            nn.MaxPool1d(8)
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 26),
            nn.Linear(26, 20)
        )

    def forward(self, x):
        x = self.CONV1(x)
        x = self.CONV2(x)
        x = self.CONV3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    # If labels are one-hot encoded, get their indices.
    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    # Get the predicted class indices for examples.
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)
    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)))  # , normalize=normalize) 部分版本无该参数
    return cmtx

# 绘图函数
def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


NIR = NIR_CONV3(1, 32, 19, 1, 0).to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(NIR.parameters(), lr=LR, weight_decay=0.01)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

if __name__ == "__main__":
    start1=time.time()
    ep = []
    trian_loss_list = []
    trian_acc = []
    test_loss_list = []
    test_acc = []
    sum_loss = 0
    train_sum_acc = 0.0
    test_sum = 0.0
    i = 0
    with open(train_result_path, "w") as f1:
        f1.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
        f1.write('\n')
        with open(test_result_path, "w") as f2:
            for epoch in range(EPOCH):
                sum_loss = 0.0  # 初始化损失度为0
                correct = 0.0  # 初始化，正确为0
                total = 0.0  # 初始化总数
                for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
                    inputs, labels = data  # 输入和标签都等于data
                    inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                    labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                    output = NIR(inputs)  # cnn output
                    loss = criterion(output, labels)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    sum_loss += loss.item()  # 每次loss相加，item 为loss转换为float
                    _, predicted = torch.max(output.data,
                                             1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 max返回两个，第一个，每行最大的概率，第二个，最大概率的索引
                    total += labels.size(0)  # 计算总的数据
                    correct += (predicted == labels).cpu().sum().data.numpy()  # 计算相等的数据
                    train_sum_acc += correct
                    sum_loss += loss.cpu().detach().numpy()
                    sum_loss = round(sum_loss, 6)
                    print("epoch = {:} Loss = {:.4f}  Acc= {:.4f}".format((epoch + 1), (loss.item()),(correct / total)))  # 训练次数，总损失，精确度
                    f1.write("{:},{:.4f},{:.4f}".format((epoch + 1), (loss.item()), (correct / total)))  # 写入数据
                    f1.write('\n')
                    f1.flush()
            end1 = time.time()
            print('training time:'+str(end1-start1))
            start2=time.time()
            with torch.no_grad():  # 无梯度
                # sum_acc = 0.0
                # for index in range(10):
                correct = 0.0  # 准确度初始化0
                total = 0.0  # 总量为0
                for data in test_loader:
                    NIR.eval()  # 不训练
                    inputs, labels = data  # 输入和标签都等于data
                    inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                    labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                    outputs = NIR(inputs)  # 输出等于进入网络后的输入
                    _, predicted = torch.max(outputs.data,
                                             1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度
                    # ，取得分最高的那个类 (outputs.data的索引号)
                    total += labels.size(0)  # 计算总的数据
                    correct += (predicted == labels).sum().cpu()  # 正确数量
                acc = 100. * correct / total
                print("Acc= {:.4f}".format(acc))  # 训练次数，总损失，精确度
                end2 = time.time()
                print('running time:'+str(end2-start2))
                '''
                torch.save(NIR, "./CNNwithAcc= {:.4f}".format(acc))
                labels = le.inverse_transform(labels)
                predicted = le.inverse_transform(predicted)
                y = le.inverse_transform(y)
                cm = confusion_matrix(labels, predicted)
                cm = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
                cm.index.name = '实际种类'
                cm.columns.name = '预测种类'
                # fig, ax = plt.subplots(figsize=(15, 10))
                plt.rcParams.update({'font.size': 7})
                cmap = "Pastel1"
                pp_matrix(cm, cmap=cmap, figsize=[15, 9])
                '''
                # plt.savefig(r"C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\SVM_CM\SVM混淆矩阵(参数C：{}P：{}).png".format(CSVC1,d))
                # 预测结果
        # 将每次测试结果实时写入acc.txt文件中
