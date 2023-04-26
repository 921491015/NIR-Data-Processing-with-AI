import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from pretty_confusion_matrix import pp_matrix
from sklearn.metrics import confusion_matrix
import joblib

# 传入SVM处理的矩阵
def SVM_data(CSVC1, d, array):
    grape_data = pd.read_csv(
        r'C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\1_vergalijo_20_varieties.csv')
    y = grape_data.iloc[:, 100:101].values  # 标签
    x = array  # 处理后的数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # 划分10%数据为测试，90%为训练
    '''
    x_test=x#这里必须改成原数据集，全部进入test，论文就是这样的
    y_test=y
    '''
    classifier = SVC(C=CSVC1, kernel='poly', degree=d)  # 设定核参数,poly是多项式
    classifier.fit(x_train, y_train)

    # 测试集结果
    y_pred = classifier.predict(x_test)
    label = grape_data.iloc[:, 100:101]
    label1 = label.drop_duplicates(subset=None, keep="first", inplace=False, ignore_index=False)
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=np.unique(label), columns=np.unique(label))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    # fig, ax = plt.subplots(figsize=(15, 10))
    plt.rcParams.update({'font.size': 7})
    cmap = "Pastel1"
    pp_matrix(cm, cmap=cmap, figsize=[15, 9])
    # plt.savefig(r"C:\Users\92149\Desktop\varietal_classification_raw_spectra_PONE-D-15-379\picture\SVM_CM\SVM混淆矩阵(参数C：{}P：{}).png".format(CSVC1,d))
    # 预测结果
    counter = cm.shape[0]
    list_diag = np.diag(cm)  # 混淆矩阵对角
    list_raw_sum = np.sum(cm, axis=1)  # 横向相加
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # 除法，后化简无穷,每个品种的准确率
    average_acc = np.mean(each_acc)  # 平均准确率
    kappa = metrics.cohen_kappa_score(y_pred, y_test)
    # overall_acc = metrics.accuracy_score(y_pred, y_test)
    CSVC2 = str(CSVC1)
    d2 = str(d)
    print('核参数：惩罚参数-' + CSVC2 + '多项式阶数-' + d2 + '的SVM\n')
    print('平均准确率:')
    print(average_acc)
    print('kappa指数:')
    print(kappa)
    joblib.dump(classifier, 'SVMwithACC={}CSVC={}.pkl'.format(average_acc, CSVC2))#保存模型
