import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, classification_report, accuracy_score, make_scorer
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from pretty_confusion_matrix import pp_matrix
import matplotlib
matplotlib.use('TkAgg')  # 绘图必须添加这个，否则出对象错误

# 传入矩阵，网络结构，学习率参数
def ANN_data(array, hl_sizes, l_rate):
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    y = grape_data.iloc[:, 100:101].values  # 标签
    y = np.array(y)
    x = array
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # 划分30%数据为测试，70%为训练
    '''
    x_train = x
    y_train = y

    x_test = x
    y_test = y
    '''
    classifier = MLPClassifier(activation='relu', hidden_layer_sizes=hl_sizes, solver='lbfgs',
                               learning_rate_init=l_rate, max_iter=3000)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    train_score = classifier.score(x_test, y_test)
    joblib.dump(classifier, 'ANNwithACC={}PARAM={}.pkl'.format(train_score, hl_sizes))

    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
    cm.index.name = '实际种类'
    cm.columns.name = '预测种类'
    plt.rcParams.update({'font.size': 7})
    cmap = "Pastel1"
    # pp_matrix(cm, cmap=cmap, figsize=[15, 9])
    print(train_score)
